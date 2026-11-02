//! Immutable segment model for the sparse-ngram index.
//!
//! # Design motivation
//!
//! `postings.bin` is a tightly-packed binary file: offsets are absolute, so a
//! single insertion would require rewriting the entire file.  The segment model
//! avoids this by treating each index snapshot as **write-once / read-many**:
//!
//! * New files are indexed into a *new* segment rather than modifying an
//!   existing one.
//! * Deleted or superseded files are recorded in a per-segment
//!   [`DeletionBitmap`] outside the immutable postings data.
//! * Queries merge results across all segments at the **path level**: the
//!   newest segment wins for any given file path.
//!
//! # Directory layout
//!
//! ```text
//! .sngi-store/                            ← store_dir (IndexStorePaths root)
//!   state.json                            – ManagerState (segment list + next_id)
//!   config.json                           – store-level config (optional)
//!   segments/
//!     segment_0/
//!       lookup.bin, postings.bin, files.txt, meta.json  (standard Index layout)
//!       deletion.bin      – DeletionBitmap for this segment
//!     segment_1/
//!       …
//!   branches/                             – per-branch manifests (future)
//! ```

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::deletion_bitmap::SegmentDeletions;
use crate::index::{Index, IndexBuilder, IndexMeta};
use crate::query::{QueryEngine, SearchResult};
use crate::storage::{IndexPaths, IndexWriter};

// ---------------------------------------------------------------------------
// CommitRange
// ---------------------------------------------------------------------------

/// The (exclusive-from, inclusive-to) commit range a segment covers.
///
/// When a segment is built from a git diff between two commits, `from` is the
/// older commit (exclusive lower bound) and `to` is the newer commit
/// (inclusive upper bound).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitRange {
    /// Exclusive lower bound (the base commit that was already indexed).
    pub from: String,
    /// Inclusive upper bound (the newest commit in this segment).
    pub to: String,
}

// ---------------------------------------------------------------------------
// SegmentLevel
// ---------------------------------------------------------------------------

/// Tier of a segment within the LSM-like index hierarchy.
///
/// | Level | Name              | Typical size | Created by          |
/// |-------|-------------------|--------------|---------------------|
/// | L1    | WAL segment       | small        | every `sync()` call |
/// | L2    | Compacted segment | medium       | merging L1s         |
/// | L3    | Base segment      | large        | full initial build  |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SegmentLevel {
    /// Small incremental segment produced by each `sync()`.
    #[default]
    L1,
    /// Medium segment produced by compacting several L1s.
    L2,
    /// Large base segment built from a complete repository tree.
    L3,
}

// ---------------------------------------------------------------------------
// SegmentStats
// ---------------------------------------------------------------------------

/// Per-segment statistics returned by [`SegmentManager::segment_stats`].
#[derive(Debug, Clone)]
pub struct SegmentStats {
    /// Total number of file slots (including deleted/tombstoned entries).
    pub total_files: usize,
    /// Number of files marked as deleted in the deletion bitmap.
    pub deleted_files: usize,
    /// Fraction of file slots that are deleted (`deleted / total`).
    pub deletion_ratio: f64,
}

// ---------------------------------------------------------------------------
// Segment
// ---------------------------------------------------------------------------

/// An opened, immutable index segment.
pub struct Segment {
    /// Monotonically increasing identifier.  Older segments have smaller ids.
    pub id: u64,
    /// Tier of this segment in the compaction hierarchy.
    pub level: SegmentLevel,
    /// The opened, ready-to-query index for this segment.
    pub index: Index,
    /// The commit range this segment covers, if built from a git history.
    pub commit_range: Option<CommitRange>,
}

// ---------------------------------------------------------------------------
// Serialisable state
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
struct SegmentRecord {
    id: u64,
    #[serde(default)]
    level: SegmentLevel,
    commit_range: Option<CommitRange>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct ManagerState {
    /// Next segment id to assign.
    next_id: u64,
    /// Ordered list of segment records (oldest first).
    segments: Vec<SegmentRecord>,
}

// ---------------------------------------------------------------------------
// SegmentManager
// ---------------------------------------------------------------------------

const STATE_FILE: &str = "state.json";
const SEGMENT_DIR_PREFIX: &str = "segment_";

/// Manages a collection of immutable [`Segment`]s stored under a single
/// directory.
///
/// Segments are kept in ascending `id` order (oldest first).  Searches
/// iterate from newest to oldest, merging results at the path level so newer
/// segments shadow older ones.
pub struct SegmentManager {
    /// Segments sorted by id ascending (index 0 = oldest).
    segments: Vec<Segment>,
    /// Per-segment deletion bitmaps (loaded on open, persisted on mutation).
    deletions: SegmentDeletions,
    /// Root directory for all segment sub-directories and `state.json`.
    store_dir: PathBuf,
    /// The next segment id to assign.
    next_id: u64,
    /// Path → list of (segment_id, file_id), used by `mark_deleted` to find
    /// every occurrence of a path.  Built lazily on first use: searches never
    /// consult it, so `open` skips the (expensive on large repos) build.
    path_index: Option<HashMap<PathBuf, Vec<(u64, u32)>>>,
}

impl SegmentManager {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new, empty manager rooted at `store_dir`.
    ///
    /// The directory (and any missing parents) is created if it does not
    /// already exist.  Any existing `state.json` is overwritten.
    pub fn create(store_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(store_dir)
            .with_context(|| format!("create segment store dir {:?}", store_dir))?;
        // Create the `segments/` subdirectory where all segment data lives.
        std::fs::create_dir_all(store_dir.join("segments"))
            .with_context(|| format!("create segments subdir in {:?}", store_dir))?;
        let mgr = Self {
            segments: Vec::new(),
            deletions: SegmentDeletions::new(),
            path_index: None,
            store_dir: store_dir.to_path_buf(),
            next_id: 0,
        };
        mgr.save_state()?;
        Ok(mgr)
    }

    /// Open an existing manager from `store_dir`.
    ///
    /// Returns an error if `store_dir` does not contain a valid `state.json`
    /// or any segment index is missing / corrupt.
    pub fn open(store_dir: &Path) -> Result<Self> {
        let state_path = store_dir.join(STATE_FILE);
        let state: ManagerState = {
            let json = std::fs::read_to_string(&state_path)
                .with_context(|| format!("read {:?}", state_path))?;
            serde_json::from_str(&json).with_context(|| format!("parse {:?}", state_path))?
        };

        let mut segments = Vec::with_capacity(state.segments.len());
        let mut deletions = SegmentDeletions::new();

        // T3: GC orphaned segment directories not in state.json
        let segments_dir = store_dir.join("segments");
        if segments_dir.exists() {
            let known_ids: std::collections::HashSet<u64> =
                state.segments.iter().map(|s| s.id).collect();
            if let Ok(entries) = std::fs::read_dir(&segments_dir) {
                for entry in entries.flatten() {
                    if let Some(name) = entry.file_name().to_str() {
                        if let Ok(id) = name.parse::<u64>() {
                            if !known_ids.contains(&id) {
                                // Orphaned directory, remove it
                                let _ = std::fs::remove_dir_all(entry.path());
                            }
                        }
                    }
                }
            }
        }

        for record in &state.segments {
            let seg_dir = segment_dir(store_dir, record.id);
            let index = Index::open(&seg_dir)
                .with_context(|| format!("open index for segment {}", record.id))?;
            deletions.load_segment(record.id, &seg_dir)?;
            segments.push(Segment {
                id: record.id,
                level: record.level,
                index,
                commit_range: record.commit_range.clone(),
            });
        }

        Ok(Self {
            segments,
            deletions,
            path_index: None,
            store_dir: store_dir.to_path_buf(),
            next_id: state.next_id,
        })
    }

    /// Build `path_index` from the live segments on first use.
    fn ensure_path_index(&mut self) {
        if self.path_index.is_some() {
            return;
        }
        let mut map: HashMap<PathBuf, Vec<(u64, u32)>> = HashMap::new();
        for segment in &self.segments {
            for (file_id, path) in segment.index.reader.file_paths.iter().enumerate() {
                if !path.as_os_str().is_empty() {
                    map.entry(path.clone())
                        .or_default()
                        .push((segment.id, file_id as u32));
                }
            }
        }
        self.path_index = Some(map);
    }

    /// Open an existing manager if `store_dir/state.json` is present,
    /// otherwise create a fresh one.
    pub fn open_or_create(store_dir: &Path) -> Result<Self> {
        if store_dir.join(STATE_FILE).exists() {
            Self::open(store_dir)
        } else {
            Self::create(store_dir)
        }
    }

    // -----------------------------------------------------------------------
    // Segment management
    // -----------------------------------------------------------------------

    /// Build a new segment from `builder`, persist it to disk, and append it
    /// to this manager.  Returns the new segment's id.
    ///
    /// `level` sets the tier of the segment in the compaction hierarchy.
    /// Incremental (`sync`) segments should use [`SegmentLevel::L1`]; full
    /// initial builds should use [`SegmentLevel::L3`].
    pub fn add_segment(
        &mut self,
        builder: IndexBuilder,
        commit_range: Option<CommitRange>,
        level: SegmentLevel,
    ) -> Result<u64> {
        let id = self.next_id;
        self.next_id += 1;

        let seg_dir = segment_dir(&self.store_dir, id);
        let index = builder
            .build(&seg_dir)
            .with_context(|| format!("build segment {id}"))?;

        // Update the path index before pushing the new segment.
        if let Some(path_index) = &mut self.path_index {
            for (file_id, file_path) in index.reader.file_paths.iter().enumerate() {
                if !file_path.as_os_str().is_empty() {
                    path_index
                        .entry(file_path.clone())
                        .or_default()
                        .push((id, file_id as u32));
                }
            }
        }

        self.segments.push(Segment {
            id,
            level,
            index,
            commit_range,
        });
        self.save_state()?;
        Ok(id)
    }

    /// Mark every occurrence of `path` across all segments as deleted.
    ///
    /// Use this to hide a path that should no longer appear in search results,
    /// such as when a file has been removed.  Ordinary updates do not require
    /// calling this method: newer segments naturally shadow older entries for
    /// the same path during newest-first search.
    pub fn mark_deleted(&mut self, path: &Path) -> Result<()> {
        self.ensure_path_index();
        if let Some(entries) = self.path_index.as_ref().and_then(|pi| pi.get(path)) {
            for &(segment_id, file_id) in entries {
                self.deletions.mark(segment_id, file_id);
            }
        }
        self.save_deletions()
    }

    /// Permanently remove segment `id` from this manager.
    ///
    /// The segment's on-disk directory is deleted, its deletion bitmap is
    /// evicted from memory, and `state.json` is updated.  All `path_index`
    /// entries that referenced this segment are pruned.
    ///
    /// Returns an error if `id` is not a known segment.
    pub fn remove_segment(&mut self, id: u64) -> Result<()> {
        self.remove_segments(&[id])
    }

    /// Permanently remove multiple segments from this manager in a single operation.
    ///
    /// This is more efficient than calling `remove_segment` in a loop, as it only
    /// persists `state.json` once at the end instead of once per segment.
    ///
    /// The segments' on-disk directories are deleted, their deletion bitmaps are
    /// evicted from memory, and `state.json` is updated once.  All `path_index`
    /// entries that referenced these segments are pruned.
    ///
    /// Returns an error if any `id` is not a known segment.
    pub fn remove_segments(&mut self, ids: &[u64]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }

        let ids_to_remove: HashSet<u64> = ids.iter().copied().collect();

        // Verify all ids exist before removing any
        for &id in ids {
            if !self.segments.iter().any(|s| s.id == id) {
                anyhow::bail!("remove_segments: segment {id} not found");
            }
        }

        // Remove all path_index entries that reference these segments.
        if let Some(path_index) = &mut self.path_index {
            for entries in path_index.values_mut() {
                entries.retain(|(seg_id, _)| !ids_to_remove.contains(seg_id));
            }
            path_index.retain(|_, v| !v.is_empty());
        }

        // Drop the Segment structs (closes the memory-mapped files) before
        // deleting the on-disk directories.
        self.segments
            .retain(|segment| !ids_to_remove.contains(&segment.id));

        // Remove the in-memory deletion bitmaps.
        for &id in ids {
            self.deletions.remove_segment(id);
        }

        // T3: Persist the updated segment list BEFORE deleting directories.
        // This ensures reopen will work even if deletion fails.
        self.save_state()?;

        // Delete the segment directories from disk.
        // Failures here are tolerated (orphaned dirs will be GC'd on next open).
        for &id in ids {
            let seg_dir = segment_dir(&self.store_dir, id);
            if seg_dir.exists() {
                let _ = std::fs::remove_dir_all(&seg_dir);
            }
        }

        Ok(())
    }

    /// Merge the segments identified by `segment_ids` into a single new segment.
    ///
    /// For every file path, only the copy from the **highest-id non-deleted**
    /// segment is included in the output (newest-wins semantics, consistent
    /// with [`Self::search`]).  The input segments are removed from disk once
    /// the new segment has been successfully written.
    ///
    /// `target_level` sets the tier of the resulting merged segment.
    ///
    /// Returns the id of the newly created compacted segment.
    ///
    /// # Errors
    ///
    /// Returns an error if `segment_ids` is empty, if any id is not managed
    /// by this instance, or if any disk operation fails.
    pub fn compact_segments(
        &mut self,
        segment_ids: &[u64],
        target_level: SegmentLevel,
    ) -> Result<u64> {
        if segment_ids.is_empty() {
            anyhow::bail!("compact_segments: segment_ids must not be empty");
        }

        // Sort ascending so `ordered[last]` is the newest segment.
        let mut ordered: Vec<u64> = segment_ids.to_vec();
        ordered.sort_unstable();

        // ── Step 1: canonical (segment_id, file_id) for each path ──────────
        // Process newest-first: `or_insert` keeps the first (newest) occurrence.
        let mut canonical: HashMap<PathBuf, (u64, u32)> = HashMap::new();
        for &seg_id in ordered.iter().rev() {
            let seg = self
                .segments
                .iter()
                .find(|s| s.id == seg_id)
                .ok_or_else(|| anyhow::anyhow!("compact_segments: segment {seg_id} not found"))?;
            for (file_id, path) in seg.index.reader.file_paths.iter().enumerate() {
                if path.as_os_str().is_empty() {
                    continue; // tombstone slot
                }
                if self.deletions.is_deleted(seg_id, file_id as u32) {
                    continue;
                }
                canonical
                    .entry(path.clone())
                    .or_insert((seg_id, file_id as u32));
            }
        }

        // ── Step 2: stable new_file_id assignment (sorted by path) ─────────
        let mut sorted: Vec<(PathBuf, u64, u32)> = canonical
            .into_iter()
            .map(|(path, (sid, fid))| (path, sid, fid))
            .collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));

        let new_file_table: Vec<PathBuf> = sorted.iter().map(|(p, _, _)| p.clone()).collect();

        // remap: (old_segment_id, old_file_id) → new_file_id
        let mut remap: HashMap<(u64, u32), u32> = HashMap::new();
        for (new_id, (_, seg_id, old_id)) in sorted.iter().enumerate() {
            remap.insert((*seg_id, *old_id), new_id as u32);
        }

        // ── Step 3: merge posting lists (F13: k-way streaming merge) ──────
        // Create iterators for each segment's postings
        let mut seg_iters: Vec<(u64, PostingIterator)> = Vec::new();
        for &seg_id in &ordered {
            let seg = self
                .segments
                .iter()
                .find(|s| s.id == seg_id)
                .ok_or_else(|| anyhow::anyhow!("compact_segments: segment {seg_id} not found"))?;
            let iter = PostingIterator::new(&seg.index, seg_id, &remap)?;
            seg_iters.push((seg_id, iter));
        }

        // Use k-way merge with min-heap to stream results
        let merged_postings = kway_merge_postings(seg_iters)?;

        // ── Step 4: write the new segment to disk ──────────────────────────
        let new_id = self.next_id;
        self.next_id += 1;

        let seg_dir = segment_dir(&self.store_dir, new_id);
        let paths = IndexPaths::new(&seg_dir);
        let mut writer = IndexWriter::new(&paths)?;
        let ngram_count = merged_postings.len();
        for (hash, ids) in &merged_postings {
            if !ids.is_empty() {
                writer.write_posting(*hash, ids)?;
            }
        }
        let live_file_count = new_file_table
            .iter()
            .filter(|p| !p.as_os_str().is_empty())
            .count();
        let meta = IndexMeta::new(live_file_count, ngram_count);
        let meta_json =
            serde_json::to_string_pretty(&meta).context("serialize compacted segment meta")?;
        writer.finish(&new_file_table, &meta_json)?;

        // Open the new segment and update in-memory structures.
        let index =
            Index::open(&seg_dir).with_context(|| format!("open compacted segment {new_id}"))?;
        if let Some(path_index) = &mut self.path_index {
            for (file_id, path) in index.reader.file_paths.iter().enumerate() {
                if !path.as_os_str().is_empty() {
                    path_index
                        .entry(path.clone())
                        .or_default()
                        .push((new_id, file_id as u32));
                }
            }
        }
        self.segments.push(Segment {
            id: new_id,
            level: target_level,
            index,
            commit_range: None,
        });

        // ── Step 5: remove the input segments ──────────────────────────────
        self.remove_segments(&ordered)?;

        Ok(new_id)
    }

    // -----------------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------------

    /// Search across all segments for files matching `pattern` (regex).
    ///
    /// Segments are searched newest-first.  A path is returned at most once
    /// (the newest segment's match wins).  Files marked deleted via
    /// [`Self::mark_deleted`] are excluded.
    ///
    /// `load_content(path)` is called for each candidate file to retrieve its
    /// bytes for exact-match verification.  Returning `None` excludes the
    /// candidate (same semantics as [`QueryEngine::search`]).
    ///
    /// F5: Accepts an optional `cancel` token for cooperative cancellation.
    pub fn search<F>(
        &self,
        pattern: &str,
        load_content: F,
        cancel: Option<&crate::CancellationToken>,
    ) -> Result<Vec<SearchResult>>
    where
        F: Fn(&Path) -> Option<Vec<u8>> + Sync,
    {
        self.search_impl(pattern, load_content, cancel, true)
    }

    /// Like [`Self::search`], but verification stops at the first match per
    /// file and `match_offsets` is empty.  Use for files-only listing.
    pub fn search_files_only<F>(
        &self,
        pattern: &str,
        load_content: F,
        cancel: Option<&crate::CancellationToken>,
    ) -> Result<Vec<SearchResult>>
    where
        F: Fn(&Path) -> Option<Vec<u8>> + Sync,
    {
        self.search_impl(pattern, load_content, cancel, false)
    }

    fn search_impl<F>(
        &self,
        pattern: &str,
        load_content: F,
        cancel: Option<&crate::CancellationToken>,
        collect_offsets: bool,
    ) -> Result<Vec<SearchResult>>
    where
        F: Fn(&Path) -> Option<Vec<u8>> + Sync,
    {
        let mut results: Vec<SearchResult> = Vec::new();
        let mut seen_paths: HashSet<PathBuf> = HashSet::new();

        // F14: Parallel search across segments using rayon
        // Segments are searched in parallel, then results are merged newest->oldest
        let segment_results: Vec<(usize, Result<Vec<SearchResult>>)> = self
            .segments
            .par_iter()
            .enumerate()
            .map(|(idx, segment)| {
                // F5: Check for cancellation before searching each segment
                if let Some(token) = cancel {
                    if token.is_cancelled() {
                        return (idx, Ok(Vec::new()));
                    }
                }

                let engine = QueryEngine::new(&segment.index);
                let seg_id = segment.id;

                // Search within this segment.  Errors are propagated to the
                // caller — swallowing them would turn e.g. an invalid pattern
                // into a misleading "0 results".
                let results = if collect_offsets {
                    engine.search(pattern, |_file_id, path| load_content(path))
                } else {
                    engine.search_files_only(pattern, |_file_id, path| load_content(path))
                };

                // Filter out deleted entries within this segment
                let filtered = results.map(|rs| {
                    rs.into_iter()
                        .filter(|result| !self.deletions.is_deleted(seg_id, result.file_id))
                        .collect()
                });

                (idx, filtered)
            })
            .collect();

        // Propagate the first segment error, if any.
        let mut merged: Vec<(usize, Vec<SearchResult>)> = Vec::with_capacity(segment_results.len());
        for (idx, res) in segment_results {
            merged.push((idx, res?));
        }
        let segment_results = merged;

        // F5: Check for cancellation before merging results
        if let Some(token) = cancel {
            if token.is_cancelled() {
                anyhow::bail!("search cancelled");
            }
        }

        // Merge results newest → oldest, deduplicating by path
        // Process in reverse order (newest first)
        for (_idx, segment_results) in segment_results.into_iter().rev() {
            for result in segment_results {
                // Skip paths already returned by a newer segment
                if seen_paths.contains(&result.path) {
                    continue;
                }
                seen_paths.insert(result.path.clone());
                results.push(result);
            }
        }

        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Number of segments currently managed.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Iterate over segments in ascending id order (oldest first).
    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// Root directory for all segment data.
    pub fn store_dir(&self) -> &Path {
        &self.store_dir
    }

    /// Read-only reference to the segment deletions tracker.
    ///
    /// Useful for code that needs to check deletion status without going
    /// through the full search path (e.g., compaction).
    pub fn deletions_ref(&self) -> &crate::deletion_bitmap::SegmentDeletions {
        &self.deletions
    }

    /// All segments at a specific level.
    pub fn segments_at_level(&self, level: SegmentLevel) -> Vec<&Segment> {
        self.segments.iter().filter(|s| s.level == level).collect()
    }

    /// Number of L1 (WAL) segments.
    pub fn l1_count(&self) -> usize {
        self.segments_at_level(SegmentLevel::L1).len()
    }

    /// Number of L2 (compacted) segments.
    pub fn l2_count(&self) -> usize {
        self.segments_at_level(SegmentLevel::L2).len()
    }

    /// Per-segment deletion statistics, or `None` if the id is unknown.
    pub fn segment_stats(&self, segment_id: u64) -> Option<SegmentStats> {
        let seg = self.segments.iter().find(|s| s.id == segment_id)?;
        let total_files = seg.index.reader.file_paths.len();
        let deleted_files = seg
            .index
            .reader
            .file_paths
            .iter()
            .enumerate()
            .filter(|(fid, path)| {
                !path.as_os_str().is_empty() && self.deletions.is_deleted(segment_id, *fid as u32)
            })
            .count();
        let deletion_ratio = if total_files == 0 {
            0.0
        } else {
            deleted_files as f64 / total_files as f64
        };
        Some(SegmentStats {
            total_files,
            deleted_files,
            deletion_ratio,
        })
    }

    /// On-disk byte size of a segment directory, or an error if the id is
    /// unknown or the directory cannot be read.
    pub fn segment_disk_size(&self, segment_id: u64) -> Result<u64> {
        if !self.segments.iter().any(|s| s.id == segment_id) {
            anyhow::bail!("segment_disk_size: segment {segment_id} not found");
        }
        let seg_dir = segment_dir(&self.store_dir, segment_id);
        let mut total: u64 = 0;
        for entry in
            std::fs::read_dir(&seg_dir).with_context(|| format!("read_dir {:?}", seg_dir))?
        {
            let entry = entry.with_context(|| format!("read_dir entry in {:?}", seg_dir))?;
            if let Ok(meta) = entry.metadata() {
                if meta.is_file() {
                    total += meta.len();
                }
            }
        }
        Ok(total)
    }

    // -----------------------------------------------------------------------
    // Persistence
    // -----------------------------------------------------------------------

    fn save_state(&self) -> Result<()> {
        let state = ManagerState {
            next_id: self.next_id,
            segments: self
                .segments
                .iter()
                .map(|s| SegmentRecord {
                    id: s.id,
                    level: s.level,
                    commit_range: s.commit_range.clone(),
                })
                .collect(),
        };
        let json =
            serde_json::to_string_pretty(&state).context("serialize SegmentManager state")?;
        let path = self.store_dir.join(STATE_FILE);
        // T9: Write state.json atomically
        crate::storage::write_atomic(&path, json.as_bytes())
            .with_context(|| format!("write {:?}", path))?;
        Ok(())
    }

    fn save_deletions(&self) -> Result<()> {
        for segment in &self.segments {
            let seg_dir = segment_dir(&self.store_dir, segment.id);
            self.deletions.save_segment(segment.id, &seg_dir)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// F13: K-way streaming compaction merge
// ---------------------------------------------------------------------------

use std::collections::BinaryHeap;

/// Iterator over posting lists from a single segment, with file_id remapping.
struct PostingIterator<'a> {
    seg_id: u64,
    index: usize,
    count: usize,
    reader: &'a crate::index::Index,
    remap: &'a HashMap<(u64, u32), u32>,
}

impl<'a> PostingIterator<'a> {
    fn new(
        index: &'a crate::index::Index,
        seg_id: u64,
        remap: &'a HashMap<(u64, u32), u32>,
    ) -> Result<Self> {
        let count = index.ngram_count() as usize;
        Ok(Self {
            seg_id,
            index: 0,
            count,
            reader: index,
            remap,
        })
    }

    fn next_posting(&mut self) -> Result<Option<(u64, Vec<u32>)>> {
        while self.index < self.count {
            let i = self.index;
            self.index += 1;

            let hash = self.reader.reader_entry_hash(i);
            let offset = self.reader.reader_entry_offset(i);
            let old_ids = self.reader.reader.read_posting_at(offset)?;

            let mut new_ids = Vec::new();
            for old_fid in old_ids {
                if let Some(&new_id) = self.remap.get(&(self.seg_id, old_fid)) {
                    new_ids.push(new_id);
                }
            }

            if !new_ids.is_empty() {
                return Ok(Some((hash, new_ids)));
            }
        }
        Ok(None)
    }
}

/// Heap entry for k-way merge: stores hash, file_ids, and iterator index.
#[derive(Eq, PartialEq)]
struct HeapEntry {
    hash: u64,
    ids: Vec<u32>,
    iter_idx: usize,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap: reverse the hash comparison
        other.hash.cmp(&self.hash)
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// K-way merge of posting lists from multiple segments using a min-heap.
/// Returns a BTreeMap of merged, deduplicated posting lists.
fn kway_merge_postings<'a>(
    mut iters: Vec<(u64, PostingIterator<'a>)>,
) -> Result<BTreeMap<u64, Vec<u32>>> {
    let mut heap = BinaryHeap::new();
    let mut result: BTreeMap<u64, Vec<u32>> = BTreeMap::new();

    // Initialize heap with first posting from each iterator
    for (idx, (_seg_id, iter)) in iters.iter_mut().enumerate() {
        if let Some((hash, ids)) = iter.next_posting()? {
            heap.push(HeapEntry {
                hash,
                ids,
                iter_idx: idx,
            });
        }
    }

    // Process entries in sorted order
    let mut current_hash: Option<u64> = None;
    let mut current_ids: Vec<u32> = Vec::new();

    while let Some(entry) = heap.pop() {
        let hash = entry.hash;
        let ids = entry.ids;
        let iter_idx = entry.iter_idx;

        // Merge ids with same hash
        if current_hash == Some(hash) {
            current_ids.extend(ids);
        } else {
            // Flush previous hash
            if let Some(h) = current_hash {
                current_ids.sort_unstable();
                current_ids.dedup();
                if !current_ids.is_empty() {
                    result.insert(h, std::mem::take(&mut current_ids));
                }
            }
            current_hash = Some(hash);
            current_ids = ids;
        }

        // Advance the iterator and push next entry
        let (_seg_id, iter) = &mut iters[iter_idx];
        if let Some((next_hash, next_ids)) = iter.next_posting()? {
            heap.push(HeapEntry {
                hash: next_hash,
                ids: next_ids,
                iter_idx,
            });
        }
    }

    // Flush final hash
    if let Some(h) = current_hash {
        current_ids.sort_unstable();
        current_ids.dedup();
        if !current_ids.is_empty() {
            result.insert(h, current_ids);
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn segment_dir(store_dir: &Path, id: u64) -> PathBuf {
    store_dir
        .join("segments")
        .join(format!("{SEGMENT_DIR_PREFIX}{id}"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_segment_builder(files: &[(&str, &[u8])]) -> Result<IndexBuilder> {
        let mut builder = IndexBuilder::with_default_weights();
        for (path, content) in files {
            builder.add_file(Path::new(path), content)?;
        }
        Ok(builder)
    }

    fn content_loader<'a>(
        files: &'a [(&'a str, &'a [u8])],
    ) -> impl Fn(&Path) -> Option<Vec<u8>> + Sync + 'a {
        move |path: &Path| {
            files
                .iter()
                .find(|(p, _)| Path::new(p) == path)
                .map(|(_, c)| c.to_vec())
        }
    }

    // -----------------------------------------------------------------------
    // CommitRange
    // -----------------------------------------------------------------------

    #[test]
    fn test_commit_range_serde() {
        let cr = CommitRange {
            from: "abc".into(),
            to: "def".into(),
        };
        let json = serde_json::to_string(&cr).unwrap();
        let cr2: CommitRange = serde_json::from_str(&json).unwrap();
        assert_eq!(cr, cr2);
    }

    // -----------------------------------------------------------------------
    // SegmentManager create / open
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_empty_and_reopen() -> Result<()> {
        let tmp = TempDir::new()?;
        {
            let mgr = SegmentManager::create(tmp.path())?;
            assert_eq!(mgr.segment_count(), 0);
        }
        // Re-open.
        let mgr2 = SegmentManager::open(tmp.path())?;
        assert_eq!(mgr2.segment_count(), 0);
        Ok(())
    }

    #[test]
    fn test_open_or_create_creates_when_missing() -> Result<()> {
        let tmp = TempDir::new()?;
        let store = tmp.path().join("store");
        let mgr = SegmentManager::open_or_create(&store)?;
        assert_eq!(mgr.segment_count(), 0);
        assert!(store.join("state.json").exists());
        Ok(())
    }

    #[test]
    fn test_open_or_create_opens_existing() -> Result<()> {
        let tmp = TempDir::new()?;
        {
            let mut mgr = SegmentManager::create(tmp.path())?;
            let b = make_segment_builder(&[("a.rs", b"fn foo() {}")])?;
            mgr.add_segment(b, None, SegmentLevel::L1)?;
        }
        let mgr2 = SegmentManager::open_or_create(tmp.path())?;
        assert_eq!(mgr2.segment_count(), 1);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // add_segment
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_segment_assigns_sequential_ids() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        let id0 = mgr.add_segment(
            make_segment_builder(&[("a.rs", b"hello")])?,
            None,
            SegmentLevel::L1,
        )?;
        let id1 = mgr.add_segment(
            make_segment_builder(&[("b.rs", b"world")])?,
            None,
            SegmentLevel::L1,
        )?;

        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(mgr.segment_count(), 2);
        Ok(())
    }

    #[test]
    fn test_add_segment_persists_and_reopen() -> Result<()> {
        let tmp = TempDir::new()?;
        {
            let mut mgr = SegmentManager::create(tmp.path())?;
            let cr = CommitRange {
                from: "aaa".into(),
                to: "bbb".into(),
            };
            mgr.add_segment(
                make_segment_builder(&[("x.rs", b"test")])?,
                Some(cr),
                SegmentLevel::L1,
            )?;
        }
        let mgr2 = SegmentManager::open(tmp.path())?;
        assert_eq!(mgr2.segment_count(), 1);
        let seg = &mgr2.segments()[0];
        assert_eq!(seg.id, 0);
        assert_eq!(
            seg.commit_range,
            Some(CommitRange {
                from: "aaa".into(),
                to: "bbb".into()
            })
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // mark_deleted
    // -----------------------------------------------------------------------

    #[test]
    fn test_mark_deleted_hides_file() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        let files: &[(&str, &[u8])] = &[("a.rs", b"fn main() {}"), ("b.rs", b"fn helper() {}")];
        mgr.add_segment(make_segment_builder(files)?, None, SegmentLevel::L1)?;

        // Delete a.rs.
        mgr.mark_deleted(Path::new("a.rs"))?;

        // Search for "fn" – only b.rs should appear.
        let results = mgr.search("fn", content_loader(files), None)?;
        let paths: Vec<&PathBuf> = results.iter().map(|r| &r.path).collect();
        assert!(
            !paths.iter().any(|p| p.as_os_str() == "a.rs"),
            "a.rs should be hidden after mark_deleted"
        );
        assert!(
            paths.iter().any(|p| p.as_os_str() == "b.rs"),
            "b.rs should still be visible"
        );
        Ok(())
    }

    #[test]
    fn test_mark_deleted_persists() -> Result<()> {
        let tmp = TempDir::new()?;
        {
            let mut mgr = SegmentManager::create(tmp.path())?;
            let files: &[(&str, &[u8])] = &[("a.rs", b"fn foo() {}")];
            mgr.add_segment(make_segment_builder(files)?, None, SegmentLevel::L1)?;
            mgr.mark_deleted(Path::new("a.rs"))?;
        }
        // Re-open; the deletion bitmap must be restored.
        let mgr2 = SegmentManager::open(tmp.path())?;
        assert!(
            mgr2.deletions.is_deleted(0, 0),
            "file_id 0 must be deleted after reload"
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // search – multi-segment merging
    // -----------------------------------------------------------------------

    #[test]
    fn test_search_single_segment() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        let files: &[(&str, &[u8])] = &[
            ("main.rs", b"fn main() { println!(\"hello\"); }"),
            ("lib.rs", b"pub fn add(a: i32, b: i32) -> i32 { a + b }"),
        ];
        mgr.add_segment(make_segment_builder(files)?, None, SegmentLevel::L1)?;

        let results = mgr.search("println", content_loader(files), None)?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, PathBuf::from("main.rs"));
        Ok(())
    }

    #[test]
    fn test_search_newest_segment_wins() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        // Segment 0: old version of a.rs with "old_fn".
        let b0 = make_segment_builder(&[("a.rs", b"fn old_fn() {}")])?;
        mgr.add_segment(b0, None, SegmentLevel::L1)?;

        // Segment 1: new version of a.rs with "new_fn".
        let b1 = make_segment_builder(&[("a.rs", b"fn new_fn() {}")])?;
        mgr.add_segment(b1, None, SegmentLevel::L1)?;

        // No mark_deleted needed here – `seen_paths` dedup in search ensures
        // that the newest segment's result shadows the older one.

        // Search for "new_fn": segment 1 (newest) should surface a.rs exactly once.
        let results = mgr.search(
            "new_fn",
            |path| {
                if path == Path::new("a.rs") {
                    Some(b"fn new_fn() {}".to_vec())
                } else {
                    None
                }
            },
            None,
        )?;
        assert_eq!(results.len(), 1, "a.rs should appear exactly once");
        assert_eq!(results[0].path, PathBuf::from("a.rs"));
        assert!(!results[0].match_offsets.is_empty());

        // Search for "old_fn": segment 0 would match, but the content loader
        // returns the *current* file content (new version), so regex verification
        // fails and no results are produced.
        let results2 = mgr.search("old_fn", |_| Some(b"fn new_fn() {}".to_vec()), None)?;
        assert!(
            results2.is_empty(),
            "old content should not match current file: {:?}",
            results2
        );
        Ok(())
    }

    #[test]
    fn test_mark_deleted_hides_all_occurrences() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        // Index "a.rs" in two segments.
        let content: &[u8] = b"fn shared() {}";
        mgr.add_segment(
            make_segment_builder(&[("a.rs", content)])?,
            None,
            SegmentLevel::L1,
        )?;
        mgr.add_segment(
            make_segment_builder(&[("a.rs", content)])?,
            None,
            SegmentLevel::L1,
        )?;

        // Mark deleted in ALL segments – simulates a true file deletion.
        mgr.mark_deleted(Path::new("a.rs"))?;

        let results = mgr.search("shared", |_| Some(content.to_vec()), None)?;
        assert!(
            results.is_empty(),
            "a.rs should be hidden in all segments: {:?}",
            results
        );
        Ok(())
    }

    #[test]
    fn test_search_dedup_by_path() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        // Same path indexed in two segments without marking deleted.
        let content: &[u8] = b"fn shared() {}";
        let b0 = make_segment_builder(&[("shared.rs", content)])?;
        let b1 = make_segment_builder(&[("shared.rs", content)])?;
        mgr.add_segment(b0, None, SegmentLevel::L1)?;
        mgr.add_segment(b1, None, SegmentLevel::L1)?;

        // Without mark_deleted the path should still appear only once.
        let results = mgr.search("shared", |_| Some(content.to_vec()), None)?;
        let count = results
            .iter()
            .filter(|r| r.path == Path::new("shared.rs"))
            .count();
        assert_eq!(count, 1, "path dedup should yield exactly one result");
        Ok(())
    }

    #[test]
    fn test_search_no_results() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;
        let files: &[(&str, &[u8])] = &[("a.rs", b"fn foo() {}")];
        mgr.add_segment(make_segment_builder(files)?, None, SegmentLevel::L1)?;

        let results = mgr.search("xyzzy_not_present_99999", content_loader(files), None)?;
        assert!(results.is_empty());
        Ok(())
    }

    #[test]
    fn test_empty_manager_search() -> Result<()> {
        let tmp = TempDir::new()?;
        let mgr = SegmentManager::create(tmp.path())?;
        let results = mgr.search("anything", |_| None, None)?;
        assert!(results.is_empty());
        Ok(())
    }

    /// Regression test: patterns that match the empty string (e.g. `.*`)
    /// must not surface deleted or already-seen files.
    #[test]
    fn test_search_empty_matching_pattern_respects_deletions() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        let content: &[u8] = b"hello world";
        mgr.add_segment(
            make_segment_builder(&[("a.rs", content)])?,
            None,
            SegmentLevel::L1,
        )?;

        // Mark the file as deleted.
        mgr.mark_deleted(Path::new("a.rs"))?;

        // `.*` matches the empty string, so any content (including empty bytes)
        // would have produced a false positive under the old `Some(vec![])` sentinel.
        let results = mgr.search(".*", |_| Some(content.to_vec()), None)?;
        assert!(
            results.is_empty(),
            "deleted file must not appear even with an empty-string-matching pattern: {:?}",
            results
        );
        Ok(())
    }

    /// Regression test: a path already seen in a newer segment must not appear
    /// again from an older segment, even when the pattern matches the empty string.
    #[test]
    fn test_search_dedup_respects_empty_matching_pattern() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        let content: &[u8] = b"hello";
        // Same path in two segments.
        mgr.add_segment(
            make_segment_builder(&[("a.rs", content)])?,
            None,
            SegmentLevel::L1,
        )?;
        mgr.add_segment(
            make_segment_builder(&[("a.rs", content)])?,
            None,
            SegmentLevel::L1,
        )?;

        let results = mgr.search(".*", |_| Some(content.to_vec()), None)?;
        let count = results
            .iter()
            .filter(|r| r.path == Path::new("a.rs"))
            .count();
        assert_eq!(
            count, 1,
            "path dedup must hold for empty-matching patterns too"
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // remove_segment
    // -----------------------------------------------------------------------

    #[test]
    fn test_remove_segment_drops_from_search() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        let content: &[u8] = b"fn unique_remove_me() {}";
        let id = mgr.add_segment(
            make_segment_builder(&[("a.rs", content)])?,
            None,
            SegmentLevel::L1,
        )?;

        // Confirm it's findable before removal.
        let before = mgr.search("unique_remove_me", |_| Some(content.to_vec()), None)?;
        assert_eq!(before.len(), 1, "should be found before remove");

        mgr.remove_segment(id)?;

        // Must be gone after removal.
        let after = mgr.search("unique_remove_me", |_| Some(content.to_vec()), None)?;
        assert!(after.is_empty(), "must not be found after remove_segment");
        Ok(())
    }

    #[test]
    fn test_remove_segment_deletes_directory() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;
        let id = mgr.add_segment(
            make_segment_builder(&[("a.rs", b"fn x() {}")])?,
            None,
            SegmentLevel::L1,
        )?;

        let seg_dir = tmp.path().join("segments").join(format!("segment_{id}"));
        assert!(seg_dir.exists(), "segment dir should exist after add");

        mgr.remove_segment(id)?;
        assert!(
            !seg_dir.exists(),
            "segment dir should be deleted after remove"
        );
        Ok(())
    }

    #[test]
    fn test_remove_segment_updates_state_json() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;
        let id = mgr.add_segment(
            make_segment_builder(&[("a.rs", b"fn x() {}")])?,
            None,
            SegmentLevel::L1,
        )?;
        mgr.remove_segment(id)?;

        // Re-open: state.json must not reference the removed segment.
        let mgr2 = SegmentManager::open(tmp.path())?;
        assert_eq!(mgr2.segment_count(), 0);
        Ok(())
    }

    #[test]
    fn test_remove_segment_clears_path_index() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;
        let id = mgr.add_segment(
            make_segment_builder(&[("a.rs", b"fn x() {}")])?,
            None,
            SegmentLevel::L1,
        )?;

        mgr.ensure_path_index();
        assert!(mgr
            .path_index
            .as_ref()
            .unwrap()
            .contains_key(Path::new("a.rs")));
        mgr.remove_segment(id)?;
        assert!(
            !mgr.path_index
                .as_ref()
                .unwrap()
                .contains_key(Path::new("a.rs")),
            "path_index must not reference removed segment"
        );
        Ok(())
    }

    #[test]
    fn test_remove_nonexistent_segment_errors() {
        let tmp = TempDir::new().unwrap();
        let mut mgr = SegmentManager::create(tmp.path()).unwrap();
        assert!(
            mgr.remove_segment(999).is_err(),
            "removing a non-existent segment must return an error"
        );
    }

    #[test]
    fn test_all_postings_roundtrip() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;
        mgr.add_segment(
            make_segment_builder(&[("a.rs", b"fn foo() { let x = 42; }")])?,
            None,
            SegmentLevel::L1,
        )?;
        let seg = &mgr.segments()[0];
        let postings = seg.index.all_postings()?;
        assert!(!postings.is_empty(), "index must have at least one posting");
        // Every hash must appear in exactly one entry.
        let hashes: Vec<u64> = postings.iter().map(|(h, _)| *h).collect();
        let deduped: std::collections::HashSet<u64> = hashes.iter().copied().collect();
        assert_eq!(hashes.len(), deduped.len(), "hashes must be unique");
        Ok(())
    }

    // -----------------------------------------------------------------------
    // compact_segments
    // -----------------------------------------------------------------------

    #[test]
    fn test_compact_merges_two_segments_into_one() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        let content_a: &[u8] = b"fn compact_alpha() {}";
        let content_b: &[u8] = b"fn compact_beta() {}";
        let id0 = mgr.add_segment(
            make_segment_builder(&[("a.rs", content_a)])?,
            None,
            SegmentLevel::L1,
        )?;
        let id1 = mgr.add_segment(
            make_segment_builder(&[("b.rs", content_b)])?,
            None,
            SegmentLevel::L1,
        )?;

        let new_id = mgr.compact_segments(&[id0, id1], SegmentLevel::L2)?;

        // Old segments must be gone.
        assert_eq!(mgr.segment_count(), 1);
        assert_eq!(mgr.segments()[0].id, new_id);

        // Both patterns must be findable after compaction.
        let ra = mgr.search("compact_alpha", |_| Some(content_a.to_vec()), None)?;
        assert_eq!(ra.len(), 1);
        assert_eq!(ra[0].path, PathBuf::from("a.rs"));

        let rb = mgr.search("compact_beta", |_| Some(content_b.to_vec()), None)?;
        assert_eq!(rb.len(), 1);
        assert_eq!(rb[0].path, PathBuf::from("b.rs"));
        Ok(())
    }

    #[test]
    fn test_compact_newest_wins_for_same_path() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        let id0 = mgr.add_segment(
            make_segment_builder(&[("a.rs", b"fn old_token() {}")])?,
            None,
            SegmentLevel::L1,
        )?;
        let id1 = mgr.add_segment(
            make_segment_builder(&[("a.rs", b"fn new_token() {}")])?,
            None,
            SegmentLevel::L1,
        )?;

        mgr.compact_segments(&[id0, id1], SegmentLevel::L2)?;

        // "new_token" must be found; "old_token" must not (content loader returns new version).
        let r_new = mgr.search("new_token", |_| Some(b"fn new_token() {}".to_vec()), None)?;
        assert_eq!(r_new.len(), 1, "new content must win");

        let r_old = mgr.search("old_token", |_| Some(b"fn new_token() {}".to_vec()), None)?;
        assert!(r_old.is_empty(), "old content must be overwritten");
        Ok(())
    }

    #[test]
    fn test_compact_excludes_deleted_files() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        let content: &[u8] = b"fn to_be_deleted_compact() {}";
        let id = mgr.add_segment(
            make_segment_builder(&[("del.rs", content)])?,
            None,
            SegmentLevel::L1,
        )?;
        mgr.mark_deleted(Path::new("del.rs"))?;

        mgr.compact_segments(&[id], SegmentLevel::L2)?;

        let r = mgr.search("to_be_deleted_compact", |_| Some(content.to_vec()), None)?;
        assert!(
            r.is_empty(),
            "deleted file must not appear after compaction"
        );
        Ok(())
    }

    #[test]
    fn test_compact_result_survives_reopen() -> Result<()> {
        let tmp = TempDir::new()?;
        {
            let mut mgr = SegmentManager::create(tmp.path())?;
            let id0 = mgr.add_segment(
                make_segment_builder(&[("persist.rs", b"fn persist_token() {}")])?,
                None,
                SegmentLevel::L1,
            )?;
            let id1 = mgr.add_segment(
                make_segment_builder(&[("other.rs", b"fn other_token() {}")])?,
                None,
                SegmentLevel::L1,
            )?;
            mgr.compact_segments(&[id0, id1], SegmentLevel::L2)?;
        }

        // Re-open from disk.
        let mgr2 = SegmentManager::open(tmp.path())?;
        assert_eq!(mgr2.segment_count(), 1, "exactly one segment after reopen");

        let r = mgr2.search(
            "persist_token",
            |_| Some(b"fn persist_token() {}".to_vec()),
            None,
        )?;
        assert_eq!(r.len(), 1, "content must survive reopen after compaction");
        Ok(())
    }

    #[test]
    fn test_compact_empty_ids_errors() {
        let tmp = TempDir::new().unwrap();
        let mut mgr = SegmentManager::create(tmp.path()).unwrap();
        assert!(
            mgr.compact_segments(&[], SegmentLevel::L2).is_err(),
            "empty segment_ids must return an error"
        );
    }

    #[test]
    fn test_compact_three_segments() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        let id0 = mgr.add_segment(
            make_segment_builder(&[("x.rs", b"fn alpha_x() {}")])?,
            None,
            SegmentLevel::L1,
        )?;
        let id1 = mgr.add_segment(
            make_segment_builder(&[("y.rs", b"fn alpha_y() {}")])?,
            None,
            SegmentLevel::L1,
        )?;
        let id2 = mgr.add_segment(
            make_segment_builder(&[("z.rs", b"fn alpha_z() {}")])?,
            None,
            SegmentLevel::L1,
        )?;

        mgr.compact_segments(&[id0, id1, id2], SegmentLevel::L2)?;
        assert_eq!(mgr.segment_count(), 1);

        for (pat, content, path) in [
            ("alpha_x", b"fn alpha_x() {}".as_ref(), "x.rs"),
            ("alpha_y", b"fn alpha_y() {}".as_ref(), "y.rs"),
            ("alpha_z", b"fn alpha_z() {}".as_ref(), "z.rs"),
        ] {
            let r = mgr.search(pat, |_| Some(content.to_vec()), None)?;
            assert_eq!(
                r.len(),
                1,
                "pattern {pat} must be found after 3-way compact"
            );
            assert_eq!(r[0].path, PathBuf::from(path));
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // L1/L2/L3 segment level creation and validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_segment_l1_level() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        let id = mgr.add_segment(
            make_segment_builder(&[("a.rs", b"fn test() {}")])?,
            None,
            SegmentLevel::L1,
        )?;

        let seg = mgr.segments().iter().find(|s| s.id == id).unwrap();
        assert_eq!(seg.level, SegmentLevel::L1);
        Ok(())
    }

    #[test]
    fn test_add_segment_l2_level() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        let id = mgr.add_segment(
            make_segment_builder(&[("a.rs", b"fn test() {}")])?,
            None,
            SegmentLevel::L2,
        )?;

        let seg = mgr.segments().iter().find(|s| s.id == id).unwrap();
        assert_eq!(seg.level, SegmentLevel::L2);
        Ok(())
    }

    #[test]
    fn test_add_segment_l3_level() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        let id = mgr.add_segment(
            make_segment_builder(&[("a.rs", b"fn test() {}")])?,
            None,
            SegmentLevel::L3,
        )?;

        let seg = mgr.segments().iter().find(|s| s.id == id).unwrap();
        assert_eq!(seg.level, SegmentLevel::L3);
        Ok(())
    }

    #[test]
    fn test_segments_at_level_filters_correctly() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        // Create segments at different levels
        let id_l1_a = mgr.add_segment(
            make_segment_builder(&[("l1_a.rs", b"fn l1_a() {}")])?,
            None,
            SegmentLevel::L1,
        )?;
        let id_l1_b = mgr.add_segment(
            make_segment_builder(&[("l1_b.rs", b"fn l1_b() {}")])?,
            None,
            SegmentLevel::L1,
        )?;
        let id_l2 = mgr.add_segment(
            make_segment_builder(&[("l2.rs", b"fn l2() {}")])?,
            None,
            SegmentLevel::L2,
        )?;
        let id_l3 = mgr.add_segment(
            make_segment_builder(&[("l3.rs", b"fn l3() {}")])?,
            None,
            SegmentLevel::L3,
        )?;

        // Check L1 segments
        let l1_segs = mgr.segments_at_level(SegmentLevel::L1);
        assert_eq!(l1_segs.len(), 2);
        let l1_ids: Vec<u64> = l1_segs.iter().map(|s| s.id).collect();
        assert!(l1_ids.contains(&id_l1_a));
        assert!(l1_ids.contains(&id_l1_b));

        // Check L2 segments
        let l2_segs = mgr.segments_at_level(SegmentLevel::L2);
        assert_eq!(l2_segs.len(), 1);
        assert_eq!(l2_segs[0].id, id_l2);

        // Check L3 segments
        let l3_segs = mgr.segments_at_level(SegmentLevel::L3);
        assert_eq!(l3_segs.len(), 1);
        assert_eq!(l3_segs[0].id, id_l3);

        Ok(())
    }

    #[test]
    fn test_l1_l2_count_accessors() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        assert_eq!(mgr.l1_count(), 0);
        assert_eq!(mgr.l2_count(), 0);

        // Add L1 segments
        mgr.add_segment(
            make_segment_builder(&[("l1_a.rs", b"fn a() {}")])?,
            None,
            SegmentLevel::L1,
        )?;
        mgr.add_segment(
            make_segment_builder(&[("l1_b.rs", b"fn b() {}")])?,
            None,
            SegmentLevel::L1,
        )?;
        mgr.add_segment(
            make_segment_builder(&[("l1_c.rs", b"fn c() {}")])?,
            None,
            SegmentLevel::L1,
        )?;

        assert_eq!(mgr.l1_count(), 3);
        assert_eq!(mgr.l2_count(), 0);

        // Add L2 segments
        mgr.add_segment(
            make_segment_builder(&[("l2_a.rs", b"fn d() {}")])?,
            None,
            SegmentLevel::L2,
        )?;
        mgr.add_segment(
            make_segment_builder(&[("l2_b.rs", b"fn e() {}")])?,
            None,
            SegmentLevel::L2,
        )?;

        assert_eq!(mgr.l1_count(), 3);
        assert_eq!(mgr.l2_count(), 2);

        Ok(())
    }

    #[test]
    fn test_segment_level_persists_after_reopen() -> Result<()> {
        let tmp = TempDir::new()?;
        {
            let mut mgr = SegmentManager::create(tmp.path())?;
            mgr.add_segment(
                make_segment_builder(&[("l1.rs", b"fn l1() {}")])?,
                None,
                SegmentLevel::L1,
            )?;
            mgr.add_segment(
                make_segment_builder(&[("l2.rs", b"fn l2() {}")])?,
                None,
                SegmentLevel::L2,
            )?;
            mgr.add_segment(
                make_segment_builder(&[("l3.rs", b"fn l3() {}")])?,
                None,
                SegmentLevel::L3,
            )?;
        }

        // Re-open and verify levels persist
        let mgr2 = SegmentManager::open(tmp.path())?;
        assert_eq!(mgr2.segment_count(), 3);
        assert_eq!(mgr2.l1_count(), 1);
        assert_eq!(mgr2.l2_count(), 1);

        let l3_segs = mgr2.segments_at_level(SegmentLevel::L3);
        assert_eq!(l3_segs.len(), 1);

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Multi-level segment search
    // -----------------------------------------------------------------------

    #[test]
    fn test_search_across_l1_l2_l3_segments() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        // Add files in different levels
        let files_l1: &[(&str, &[u8])] = &[("l1_file.rs", b"fn l1_function() {}")];
        let files_l2: &[(&str, &[u8])] = &[("l2_file.rs", b"fn l2_function() {}")];
        let files_l3: &[(&str, &[u8])] = &[("l3_file.rs", b"fn l3_function() {}")];

        mgr.add_segment(make_segment_builder(files_l1)?, None, SegmentLevel::L1)?;
        mgr.add_segment(make_segment_builder(files_l2)?, None, SegmentLevel::L2)?;
        mgr.add_segment(make_segment_builder(files_l3)?, None, SegmentLevel::L3)?;

        // Search for "function" should find all three files
        let all_files = [files_l1, files_l2, files_l3].concat();
        let results = mgr.search("function", content_loader(&all_files), None)?;

        assert_eq!(results.len(), 3);
        let paths: Vec<&PathBuf> = results.iter().map(|r| &r.path).collect();
        assert!(paths.iter().any(|p| p.as_os_str() == "l1_file.rs"));
        assert!(paths.iter().any(|p| p.as_os_str() == "l2_file.rs"));
        assert!(paths.iter().any(|p| p.as_os_str() == "l3_file.rs"));

        Ok(())
    }

    #[test]
    fn test_compact_l1_to_l2_preserves_search() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        let files: &[(&str, &[u8])] = &[("a.rs", b"fn alpha() {}"), ("b.rs", b"fn beta() {}")];

        // Create two L1 segments
        let id0 = mgr.add_segment(make_segment_builder(&[files[0]])?, None, SegmentLevel::L1)?;
        let id1 = mgr.add_segment(make_segment_builder(&[files[1]])?, None, SegmentLevel::L1)?;

        // Search before compaction
        let before_alpha = mgr.search("alpha", content_loader(files), None)?;
        let before_beta = mgr.search("beta", content_loader(files), None)?;
        assert_eq!(before_alpha.len(), 1);
        assert_eq!(before_beta.len(), 1);

        // Compact L1 → L2
        mgr.compact_segments(&[id0, id1], SegmentLevel::L2)?;

        // Verify the segment is now L2
        assert_eq!(mgr.l1_count(), 0);
        assert_eq!(mgr.l2_count(), 1);

        // Search after compaction - results should be identical
        let after_alpha = mgr.search("alpha", content_loader(files), None)?;
        let after_beta = mgr.search("beta", content_loader(files), None)?;
        assert_eq!(after_alpha.len(), 1);
        assert_eq!(after_beta.len(), 1);
        assert_eq!(after_alpha[0].path, PathBuf::from("a.rs"));
        assert_eq!(after_beta[0].path, PathBuf::from("b.rs"));

        Ok(())
    }
}
