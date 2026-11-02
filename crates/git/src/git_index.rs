//! Git-aware index: segment-based on-disk store + in-memory overlay.
//!
//! # Architecture
//!
//! ```text
//! .sngi-store/                 ← store_dir (IndexStorePaths root)
//!   state.json                 ← segment list + next_id
//!   head.txt                   ← latest fully-indexed commit SHA
//!   segments/
//!     segment_0/               ← full index built from initial HEAD
//!       lookup.bin, postings.bin, files.txt, meta.json, [deletion.bin]
//!     segment_1/               ← incremental: changes from commit A → B
//!       …
//! ```
//!
//! The `OverlayIndex` (in-memory) remains for dirty working-tree changes
//! that have not been committed.  On restart the overlay is empty and
//! `sync()` re-populates it.
//!
//! # Compaction
//!
//! After every `sync()` that adds a segment, if the total number of segments
//! reaches [`COMPACT_THRESHOLD`] they are merged into one via
//! [`SegmentManager::compact_segments`].  This keeps lookup overhead bounded.
//!
//! # Lifecycle
//!
//! ```text
//!  open_or_build()
//!    ├─ state.json exists?
//!    │     └─ open SegmentManager, read head.txt, call sync()
//!    └─ otherwise
//!          └─ build full segment from HEAD tree, write head.txt
//!
//!  sync()
//!    ├─ diff HEAD vs indexed_commit → changed files → new segment
//!    ├─ mark permanently-deleted files in the store
//!    ├─ compact if segment count ≥ COMPACT_THRESHOLD
//!    ├─ re-compute dirty overlay from HEAD vs working tree
//!    └─ update head.txt
//!
//!  search(pattern)
//!    ├─ store.search()     → base + incremental segments (disk)
//!    ├─ subtract overlay-deleted / overlay-modified paths
//!    ├─ overlay candidates → regex-verify
//!    └─ merge results
//! ```

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use git2::{Delta, DiffOptions, Oid, Repository};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sparse_ngram_core::compaction::CompactionEngine;
use sparse_ngram_core::default_weight_fn;
use sparse_ngram_core::file_ngram_hashes;
use sparse_ngram_core::index::IndexBuilder;
use sparse_ngram_core::query::SearchResult;
use sparse_ngram_core::regex_decompose::{decompose_regex, QueryAtom};
use sparse_ngram_core::segment::{CommitRange, SegmentLevel, SegmentManager};
use sparse_ngram_core::storage::{resolve_index_dir, should_skip_dir};

use crate::overlay::OverlayIndex;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// File name (relative to the store dir) that stores the latest indexed commit.
const HEAD_FILE: &str = "head.txt";

/// File name (relative to the store dir) of the persisted dirty-scan cache.
const DIRTY_SCAN_FILE: &str = "dirty_scan.json";

/// Upper bound of the adaptive dirty-scan TTL (milliseconds).
///
/// The effective TTL is `min(last_scan_duration × 10, MAX_ADAPTIVE_TTL_MS)`:
/// repositories where the working-tree scan is cheap stay effectively
/// always-fresh, while on very large repositories (where a scan costs
/// seconds) repeated searches skip the rescan for a bounded window.
/// `SNGS_SYNC_TTL_SECS` overrides this with a fixed TTL (0 = always rescan).
const MAX_ADAPTIVE_TTL_MS: u64 = 30_000;

// ---------------------------------------------------------------------------
// DirtyScanCache
// ---------------------------------------------------------------------------

/// Persisted result of a working-tree dirty scan.  Kept in
/// `.sngi-store/dirty_scan.json` so that repeated `sync_if_stale` calls
/// (e.g. every `sngs search`) can skip the expensive full scan when nothing
/// relevant could have changed within the TTL window.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DirtyScanCache {
    /// HEAD/indexed commit the scan was taken against.
    commit: String,
    /// Unix timestamp (seconds) when the scan completed.
    timestamp: u64,
    /// How long the scan took (milliseconds); drives the adaptive TTL.
    scan_ms: u64,
    /// `(path, is_deleted)` pairs as reported by the scan.
    files: Vec<(PathBuf, bool)>,
}

impl DirtyScanCache {
    fn load(path: &Path) -> Option<Self> {
        let s = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&s).ok()
    }

    fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string(self)?;
        sparse_ngram_core::storage::write_atomic(path, json.as_bytes())
            .with_context(|| format!("write {:?}", path))
    }
}

/// Read the optional fixed dirty-scan TTL from `SNGS_SYNC_TTL_SECS`.
/// Unset or unparsable means "use the adaptive TTL"; 0 means "always rescan".
fn dirty_scan_ttl_from_env() -> Option<Duration> {
    std::env::var("SNGS_SYNC_TTL_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map(Duration::from_secs)
}

fn unix_timestamp_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// SyncStats
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct SyncStats {
    pub added: usize,
    pub modified: usize,
    pub deleted: usize,
    /// True if a full rebuild was triggered.
    pub rebuilt: bool,
    /// True if segments were compacted during this sync.
    pub compacted: bool,
}

// ---------------------------------------------------------------------------
// GitAwareIndex
// ---------------------------------------------------------------------------

/// Segment-based on-disk index for a git repository + in-memory overlay for
/// dirty working-tree changes.
pub struct GitAwareIndex {
    /// Multi-segment on-disk index store.
    store: SegmentManager,
    /// The most recent commit that has been fully incorporated into the store.
    indexed_commit: String,
    /// In-memory overlay for uncommitted working-tree changes.
    overlay: OverlayIndex,
    /// Path to the repository root.
    repo_path: PathBuf,
    /// Compaction engine for LSM-style tiered merging.
    compaction_engine: CompactionEngine,
    /// Fixed dirty-scan TTL override (`None` = adaptive TTL).
    dirty_scan_ttl: Option<Duration>,
}

impl GitAwareIndex {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Open an existing git-aware index in **read-only** mode.
    ///
    /// F7: This method opens the store without calling sync(), making it suitable
    /// for read operations like status, search, and warmup. It does not modify
    /// the index or write to disk.
    ///
    /// Returns an error if no valid index exists.
    pub fn open_readonly(repo_path: &Path) -> Result<Self> {
        let store_dir = resolve_index_dir(repo_path);
        let head_file = store_dir.join(HEAD_FILE);

        if !store_dir.join("state.json").exists() || !head_file.exists() {
            anyhow::bail!(
                "no index found at {:?}; run `sngs init` or `sngs sync` first",
                store_dir
            );
        }

        let indexed_commit = std::fs::read_to_string(&head_file)
            .with_context(|| format!("read {:?}", head_file))?
            .trim()
            .to_string();

        let store = SegmentManager::open(&store_dir)
            .with_context(|| format!("open segment store at {:?}", store_dir))?;

        Ok(Self {
            store,
            indexed_commit,
            overlay: OverlayIndex::new(),
            repo_path: repo_path.to_path_buf(),
            compaction_engine: CompactionEngine::default(),
            dirty_scan_ttl: dirty_scan_ttl_from_env(),
        })
    }

    /// Open an existing segmented index, or build one from scratch.
    ///
    /// If the store directory contains a `state.json` it is opened and
    /// `sync()` is called to bring it up to date.  Otherwise a fresh full
    /// index is built from HEAD.
    pub fn open_or_build(repo_path: &Path) -> Result<Self> {
        relax_odb_hash_verification();
        let store_dir = resolve_index_dir(repo_path);
        let head_file = store_dir.join(HEAD_FILE);

        if store_dir.join("state.json").exists() && head_file.exists() {
            let indexed_commit = std::fs::read_to_string(&head_file)
                .with_context(|| format!("read {:?}", head_file))?
                .trim()
                .to_string();

            let store = SegmentManager::open(&store_dir)
                .with_context(|| format!("open segment store at {:?}", store_dir))?;

            let mut gi = Self {
                store,
                indexed_commit,
                overlay: OverlayIndex::new(),
                repo_path: repo_path.to_path_buf(),
                compaction_engine: CompactionEngine::default(),
                dirty_scan_ttl: dirty_scan_ttl_from_env(),
            };
            // Refresh within the dirty-scan TTL window: the expensive full
            // working-tree scan is skipped when the cache is still fresh.
            gi.sync_if_stale()?;
            return Ok(gi);
        }

        // No valid segmented index found — build from HEAD.
        let head_str = get_head_commit_str(repo_path)?;
        let head_oid =
            Oid::from_str(&head_str).with_context(|| format!("parse HEAD oid: {head_str}"))?;
        Self::build_from_head(repo_path, head_oid, &head_str)
    }

    /// Build a fresh index from the HEAD commit tree.
    pub fn build_from_head(repo_path: &Path, head_oid: Oid, head_str: &str) -> Result<Self> {
        relax_odb_hash_verification();
        let store_dir = resolve_index_dir(repo_path);

        let mut builder = IndexBuilder::new(default_weight_fn());

        // Open repo, collect files, close before building index.
        {
            let repo = Repository::open(repo_path)
                .with_context(|| format!("open git repo at {:?}", repo_path))?;
            let commit = repo
                .find_commit(head_oid)
                .with_context(|| "find HEAD commit")?;
            let tree = commit.tree()?;
            collect_tree_files(&tree, repo_path, &mut builder)?;
        }

        let mut store = SegmentManager::create(&store_dir)?;
        store.add_segment(
            builder,
            Some(CommitRange {
                from: head_str.to_string(),
                to: head_str.to_string(),
            }),
            SegmentLevel::L3,
        )?;

        // Write the latest indexed commit atomically.
        let head_file = store_dir.join(HEAD_FILE);
        sparse_ngram_core::storage::write_atomic(&head_file, head_str.as_bytes())
            .with_context(|| format!("write {:?}", head_file))?;

        Ok(Self {
            store,
            indexed_commit: head_str.to_string(),
            overlay: OverlayIndex::new(),
            repo_path: repo_path.to_path_buf(),
            compaction_engine: CompactionEngine::default(),
            dirty_scan_ttl: dirty_scan_ttl_from_env(),
        })
    }

    // -----------------------------------------------------------------------
    // Sync
    // -----------------------------------------------------------------------

    /// Bring the index up to date with the current HEAD and working tree.
    ///
    /// **Committed changes** (new commits since `indexed_commit`) are
    /// persisted as a new segment in the store.  **Dirty changes**
    /// (uncommitted working-tree modifications) are reflected in the
    /// in-memory overlay.
    ///
    /// This always performs a **full** working-tree scan.  Callers that sync
    /// on every operation (e.g. `sngs search`) should use
    /// [`Self::sync_if_stale`], which reuses the persisted dirty-scan cache
    /// within its TTL window.
    ///
    /// After adding a new segment, tiered compaction is triggered using the
    /// [`CompactionEngine`] to maintain LSM-style segment organization.
    pub fn sync(&mut self) -> Result<SyncStats> {
        self.sync_internal(true)
    }

    /// Like [`Self::sync`], but the working-tree dirty scan is skipped when
    /// the persisted scan cache ([`.sngi-store/dirty_scan.json`]) is still
    /// within its TTL window.
    ///
    /// The TTL is adaptive by default — `min(last_scan_duration × 10, 30s)`,
    /// so repositories with a cheap scan stay effectively always-fresh — and
    /// can be pinned with `SNGS_SYNC_TTL_SECS` (0 = always rescan).
    ///
    /// Files that become dirty *after* the cached scan may be missed until
    /// the window expires or [`Self::sync`] runs; files already present in
    /// the cached list are re-read and re-validated, so they always reflect
    /// the current working tree.
    pub fn sync_if_stale(&mut self) -> Result<SyncStats> {
        self.sync_internal(false)
    }

    /// Override the dirty-scan TTL (mainly for tests).  `Duration::ZERO`
    /// disables the cache (always rescan).
    pub fn set_dirty_scan_ttl(&mut self, ttl: Duration) {
        self.dirty_scan_ttl = Some(ttl);
    }

    fn sync_internal(&mut self, force_dirty_scan: bool) -> Result<SyncStats> {
        let head_str = get_head_commit_str(&self.repo_path)?;
        let mut stats = SyncStats::default();

        // ── Part 1: committed changes → new segment ─────────────────────────
        if head_str != self.indexed_commit {
            let changes =
                get_commit_diff_changes(&self.repo_path, &self.indexed_commit, &head_str)?;

            let mut added_builder = IndexBuilder::new(default_weight_fn());
            let mut has_additions = false;

            // Open repo once for all file reads
            let repo = Repository::open(&self.repo_path)
                .with_context(|| format!("open git repo at {:?}", self.repo_path))?;
            let head_oid =
                Oid::from_str(&head_str).with_context(|| format!("parse HEAD oid: {head_str}"))?;
            let commit = repo
                .find_commit(head_oid)
                .with_context(|| "find HEAD commit")?;
            let tree = commit.tree()?;

            for (path, is_deleted) in &changes {
                if path_in_skipped_dir(path) {
                    continue;
                }
                if *is_deleted {
                    self.store.mark_deleted(path)?;
                    stats.deleted += 1;
                } else {
                    // Read file content from the head commit tree, not the working tree.
                    if let Ok(content) = read_file_from_tree(&repo, &tree, path) {
                        // Mark any previous occurrences of this path as
                        // superseded before adding the updated content.
                        self.store.mark_deleted(path)?;
                        added_builder.add_file(path, &content)?;
                        has_additions = true;
                        stats.added += 1;
                    }
                }
            }

            if has_additions {
                self.store.add_segment(
                    added_builder,
                    Some(CommitRange {
                        from: self.indexed_commit.clone(),
                        to: head_str.clone(),
                    }),
                    SegmentLevel::L1,
                )?;
            }

            self.indexed_commit = head_str.clone();

            // Persist the updated head commit atomically.
            let head_file = self.store.store_dir().join(HEAD_FILE);
            sparse_ngram_core::storage::write_atomic(&head_file, self.indexed_commit.as_bytes())
                .with_context(|| format!("write {:?}", head_file))?;

            // Use the compaction engine to determine if and what to compact.
            if let Some(plan) = self.compaction_engine.should_compact(&self.store) {
                // Load content from the committed tree for segment rebuilds.
                let load_content = |path: &Path| -> Option<Vec<u8>> {
                    let repo = Repository::open(&self.repo_path).ok()?;
                    let head_oid = Oid::from_str(&self.indexed_commit).ok()?;
                    let commit = repo.find_commit(head_oid).ok()?;
                    let tree = commit.tree().ok()?;
                    read_file_from_tree(&repo, &tree, path).ok()
                };

                self.compaction_engine
                    .compact_segments(&mut self.store, &plan, &load_content)?;
                stats.compacted = true;
            }
        }

        // ── Part 2: dirty working-tree changes → overlay ────────────────────
        self.overlay.clear();
        let dirty = self.dirty_files(force_dirty_scan)?;
        for (path, is_deleted) in &dirty {
            if path_in_skipped_dir(path) {
                continue;
            }
            if *is_deleted {
                self.overlay.remove_file(path);
                stats.deleted += 1;
            } else {
                let full = self.repo_path.join(path);
                // Skip binary files (by extension; content is NUL-checked below).
                if is_likely_binary(path) {
                    continue;
                }
                match std::fs::read(&full) {
                    Ok(content) => {
                        // Skip binary content (NUL byte check); no size cap —
                        // the base index has no size limit either, so capping
                        // here would make large dirty files lose their newly
                        // added content from search results.
                        if content.contains(&0u8) {
                            continue;
                        }
                        self.overlay.add_file(path, &content)?;
                        stats.modified += 1;
                    }
                    Err(_) => {
                        // The file vanished between the dirty scan and the
                        // overlay rebuild (possible too for cache-hit entries
                        // deleted since the scan): hide any stale base result.
                        self.overlay.remove_file(path);
                        stats.deleted += 1;
                    }
                }
            }
        }

        Ok(stats)
    }

    /// Return the `(path, is_deleted)` dirty working-tree set, reusing the
    /// persisted scan cache when `force` is false and the cache is fresh.
    fn dirty_files(&self, force: bool) -> Result<Vec<(PathBuf, bool)>> {
        let cache_path = self.store.store_dir().join(DIRTY_SCAN_FILE);
        if !force {
            if let Some(cache) = DirtyScanCache::load(&cache_path) {
                if cache.commit == self.indexed_commit && self.cache_is_fresh(&cache) {
                    return Ok(cache.files);
                }
            }
        }

        let start = Instant::now();
        let files = get_dirty_files(&self.repo_path, &self.indexed_commit)?;
        let cache = DirtyScanCache {
            commit: self.indexed_commit.clone(),
            timestamp: unix_timestamp_secs(),
            scan_ms: start.elapsed().as_millis() as u64,
            files: files.clone(),
        };
        // Best-effort: failing to persist the cache only costs performance.
        let _ = cache.save(&cache_path);
        Ok(files)
    }

    /// Whether the cached dirty scan is still within its TTL window.
    fn cache_is_fresh(&self, cache: &DirtyScanCache) -> bool {
        let ttl = match self.dirty_scan_ttl {
            Some(d) => d,
            None => {
                Duration::from_millis(cache.scan_ms.saturating_mul(10).min(MAX_ADAPTIVE_TTL_MS))
            }
        };
        if ttl.is_zero() {
            return false;
        }
        let age = unix_timestamp_secs().saturating_sub(cache.timestamp);
        age <= ttl.as_secs()
    }

    // -----------------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------------

    /// Search for files matching `pattern` (regex or literal).
    pub fn search(
        &self,
        pattern: &str,
        cancel: Option<&sparse_ngram_core::CancellationToken>,
    ) -> Result<Vec<SearchResult>> {
        self.search_impl(pattern, cancel, true)
    }

    /// Like [`Self::search`], but verification stops at the first match per
    /// file and `match_offsets` is empty.  Use for files-only listing.
    pub fn search_files_only(
        &self,
        pattern: &str,
        cancel: Option<&sparse_ngram_core::CancellationToken>,
    ) -> Result<Vec<SearchResult>> {
        self.search_impl(pattern, cancel, false)
    }

    fn search_impl(
        &self,
        pattern: &str,
        cancel: Option<&sparse_ngram_core::CancellationToken>,
        collect_offsets: bool,
    ) -> Result<Vec<SearchResult>> {
        let repo_workdir = self.repo_path.clone();
        let weight_fn = default_weight_fn();
        let atom = decompose_regex(pattern, weight_fn.as_ref())?;

        // ── Segment store candidates (disk) ───────────────────────────────
        let overlay_ref = &self.overlay;
        let store_loader = |path: &std::path::Path| -> Option<Vec<u8>> {
            // Exclude files superseded or deleted in the overlay: their
            // store-side content is stale, and the overlay pass below
            // re-verifies them against the working tree.  Returning None
            // drops the candidate entirely (no verification possible).
            if overlay_ref.is_deleted(path) || overlay_ref.is_modified(path) {
                return None;
            }
            std::fs::read(repo_workdir.join(path)).ok()
        };
        let store_results = if collect_offsets {
            self.store
                .search(pattern, |path| store_loader(path), cancel)?
        } else {
            self.store
                .search_files_only(pattern, |path| store_loader(path), cancel)?
        };

        // ── Overlay candidates (in-memory dirty files) ────────────────────
        let overlay_paths = self.resolve_overlay_atom(&atom);
        let re = regex::bytes::Regex::new(pattern)
            .with_context(|| format!("compile regex: {pattern}"))?;

        let mut seen: HashSet<PathBuf> = store_results.iter().map(|r| r.path.clone()).collect();
        let mut results = store_results;

        for rel_path in overlay_paths {
            if seen.contains(&rel_path) {
                continue;
            }
            let full = self.repo_path.join(&rel_path);
            if let Ok(content) = std::fs::read(&full) {
                if !collect_offsets {
                    if re.is_match(&content) {
                        seen.insert(rel_path.clone());
                        results.push(SearchResult {
                            file_id: u32::MAX,
                            path: rel_path,
                            match_offsets: vec![],
                        });
                    }
                    continue;
                }
                let offsets: Vec<usize> = re
                    .find_iter(&content)
                    .map(|m: regex::bytes::Match| m.start())
                    .collect();
                if !offsets.is_empty() {
                    seen.insert(rel_path.clone());
                    results.push(SearchResult {
                        file_id: u32::MAX,
                        path: rel_path,
                        match_offsets: offsets,
                    });
                }
            }
        }

        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Status
    // -----------------------------------------------------------------------

    /// The latest commit SHA that has been incorporated into the on-disk store.
    pub fn base_commit(&self) -> &str {
        &self.indexed_commit
    }

    pub fn overlay_modified_count(&self) -> usize {
        self.overlay.modified_paths().count()
    }

    pub fn overlay_deleted_count(&self) -> usize {
        self.overlay.deleted_paths().count()
    }

    /// Number of on-disk segments in the store.
    pub fn segment_count(&self) -> usize {
        self.store.segment_count()
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    fn resolve_overlay_atom(&self, atom: &QueryAtom) -> Vec<PathBuf> {
        match atom {
            QueryAtom::Any => self
                .overlay
                .modified_paths()
                .map(|p| p.to_path_buf())
                .collect(),
            QueryAtom::And(hashes) => self.overlay.lookup_and(hashes).into_iter().collect(),
            QueryAtom::Or(branches) => {
                let sets: Vec<Vec<u64>> = branches
                    .iter()
                    .filter_map(|b| {
                        if let QueryAtom::And(h) = b {
                            Some(h.clone())
                        } else {
                            None
                        }
                    })
                    .collect();
                self.overlay.lookup_or(&sets).into_iter().collect()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Git helpers (open repo, do work, drop repo)
// ---------------------------------------------------------------------------

/// Return the HEAD commit as a hex string.
fn get_head_commit_str(repo_path: &Path) -> Result<String> {
    let repo =
        Repository::open(repo_path).with_context(|| format!("open git repo at {:?}", repo_path))?;
    let head = repo.head().context("get HEAD")?;
    let oid = head.peel_to_commit().context("peel HEAD to commit")?.id();
    Ok(oid.to_string())
}

/// Return `(relative_path, is_deleted)` pairs for files that changed between
/// `base_commit_str` and `head_str`.
fn get_commit_diff_changes(
    repo_path: &Path,
    base_str: &str,
    head_str: &str,
) -> Result<Vec<(PathBuf, bool)>> {
    let repo = Repository::open(repo_path)?;
    let base_oid = Oid::from_str(base_str)?;
    let head_oid = Oid::from_str(head_str)?;
    let base_commit = repo.find_commit(base_oid)?;
    let head_commit = repo.find_commit(head_oid)?;
    let base_tree = base_commit.tree()?;
    let head_tree = head_commit.tree()?;

    let mut opts = DiffOptions::new();
    let diff = repo.diff_tree_to_tree(Some(&base_tree), Some(&head_tree), Some(&mut opts))?;

    let mut changes = Vec::new();
    for delta in diff.deltas() {
        match delta.status() {
            Delta::Added | Delta::Modified | Delta::Renamed | Delta::Copied => {
                if let Some(p) = delta.new_file().path() {
                    changes.push((p.to_path_buf(), false));
                }
            }
            Delta::Typechange => {
                // file↔symlink transitions: drop the old entry and re-index
                // the new one (sync marks the path deleted before re-adding).
                if let Some(p) = delta.old_file().path() {
                    changes.push((p.to_path_buf(), true));
                }
                if let Some(p) = delta.new_file().path() {
                    changes.push((p.to_path_buf(), false));
                }
            }
            Delta::Deleted => {
                if let Some(p) = delta.old_file().path() {
                    changes.push((p.to_path_buf(), true));
                }
            }
            _ => {}
        }
    }
    Ok(changes)
}

/// Return `(relative_path, is_deleted)` for files that are dirty in the
/// working tree relative to `base_commit_str`.
/// F8: Includes untracked files (not in .gitignore).
fn get_dirty_files(repo_path: &Path, base_str: &str) -> Result<Vec<(PathBuf, bool)>> {
    let repo = Repository::open(repo_path)?;
    let base_oid = Oid::from_str(base_str)?;
    let commit = repo.find_commit(base_oid)?;
    let tree = commit.tree()?;

    let mut opts = DiffOptions::new();
    // F8: Include untracked files and recurse into untracked directories
    opts.include_untracked(true);
    opts.recurse_untracked_dirs(true);
    let diff = repo.diff_tree_to_workdir_with_index(Some(&tree), Some(&mut opts))?;

    let mut changes = Vec::new();
    for delta in diff.deltas() {
        match delta.status() {
            Delta::Modified | Delta::Added | Delta::Renamed | Delta::Copied | Delta::Untracked => {
                if let Some(p) = delta.new_file().path() {
                    changes.push((p.to_path_buf(), false));
                }
            }
            Delta::Typechange => {
                // file↔symlink transitions: hide the old entry, then index
                // the new one (entries are applied in order).
                if let Some(p) = delta.old_file().path() {
                    changes.push((p.to_path_buf(), true));
                }
                if let Some(p) = delta.new_file().path() {
                    changes.push((p.to_path_buf(), false));
                }
            }
            Delta::Deleted => {
                if let Some(p) = delta.old_file().path() {
                    changes.push((p.to_path_buf(), true));
                }
            }
            _ => {}
        }
    }
    Ok(changes)
}

/// Turn off libgit2's strict object hash verification.
///
/// libgit2 1.8 enables it by default, which re-hashes (SHA1DC) every object
/// read from the ODB — roughly 10% of a full index build on a large
/// repository.  The git CLI does not re-hash objects on read either; object
/// integrity is still guaranteed by the pack index/checksums at fetch time.
fn relax_odb_hash_verification() {
    git2::opts::strict_hash_verification(false);
}

/// Walk a git tree and add all text files to an `IndexBuilder`.
///
/// Pipeline:
/// 1. Walk the tree (cheap, no blob inflation) and collect `(path, oid)` for
///    every indexable blob.
/// 2. Inflate blobs and extract n-gram hash sets **in parallel** (Rayon);
///    each worker thread lazily opens its own `Repository` handle because
///    `Repository` is not `Sync`.
/// 3. Merge the per-file hash sets via [`IndexBuilder::from_file_hashes`]
///    (parallel chunk merge).
fn collect_tree_files(
    tree: &git2::Tree,
    repo_root: &Path,
    builder: &mut IndexBuilder,
) -> Result<()> {
    // ── Phase 1: collect candidate blob entries ─────────────────────────────
    let mut entries: Vec<(PathBuf, Oid)> = Vec::new();
    tree.walk(git2::TreeWalkMode::PreOrder, |root, entry| {
        use git2::ObjectType;
        // Skip entire directories that should never be indexed.
        if entry.kind() == Some(ObjectType::Tree) {
            if let Some(name) = entry.name() {
                if should_skip_dir(name) {
                    return git2::TreeWalkResult::Skip;
                }
            }
        }
        if entry.kind() != Some(ObjectType::Blob) {
            return git2::TreeWalkResult::Ok;
        }
        let rel_path = if root.is_empty() {
            PathBuf::from(entry.name().unwrap_or(""))
        } else {
            PathBuf::from(root).join(entry.name().unwrap_or(""))
        };

        if is_likely_binary(&rel_path) {
            return git2::TreeWalkResult::Ok;
        }

        entries.push((rel_path, entry.id()));
        git2::TreeWalkResult::Ok
    })
    .context("walk tree")?;

    // ── Phase 2: inflate blobs + extract n-gram hashes in parallel ──────────
    // Concrete FrequencyWeight so weight lookups are inlined (monomorphised).
    let wf = sparse_ngram_core::FrequencyWeight::with_utf8_smoothing();
    let (paths, oids): (Vec<PathBuf>, Vec<Oid>) = entries.into_iter().unzip();
    let per_file: Vec<Option<Vec<u64>>> = oids
        .par_iter()
        .map_init(
            || Repository::open(repo_root).ok(),
            |repo, oid| {
                let blob = repo.as_ref()?.find_blob(*oid).ok()?;
                let content = blob.content();
                if content.contains(&0u8) {
                    return None;
                }
                Some(file_ngram_hashes(content, &wf))
            },
        )
        .collect();

    // ── Phase 3: merge into the inverted index (parallel chunk merge) ───────
    *builder = IndexBuilder::from_file_hashes(&paths, per_file, default_weight_fn());
    Ok(())
}

/// Read the content of a file from a git tree.
///
/// Returns the file content as bytes, or an error if the file is not found,
/// is not a blob, or contains binary content.
fn read_file_from_tree(repo: &Repository, tree: &git2::Tree, path: &Path) -> Result<Vec<u8>> {
    let entry = tree
        .get_path(path)
        .with_context(|| format!("file not found in tree: {}", path.display()))?;

    let obj = entry
        .to_object(repo)
        .with_context(|| format!("resolve tree entry for {}", path.display()))?;

    let blob = obj
        .as_blob()
        .ok_or_else(|| anyhow::anyhow!("not a blob: {}", path.display()))?;

    let content = blob.content();
    if content.contains(&0u8) {
        anyhow::bail!("binary content: {}", path.display());
    }

    Ok(content.to_vec())
}

/// Return `true` if any directory component of `path` is in the always-skip list.
fn path_in_skipped_dir(path: &Path) -> bool {
    path.parent().is_some_and(|parent| {
        parent
            .components()
            .any(|c| c.as_os_str().to_str().map(should_skip_dir).unwrap_or(false))
    })
}

fn is_likely_binary(path: &Path) -> bool {
    // Keep this list limited to formats that are virtually always binary;
    // text formats (e.g. `svg`, `lock`) must stay indexable — the NUL-byte
    // content check is the real binary detector.
    let binary_exts = [
        "png", "jpg", "jpeg", "gif", "bmp", "ico", "pdf", "zip", "tar", "gz", "bz2", "xz", "7z",
        "rar", "exe", "dll", "so", "dylib", "a", "o", "wasm", "bin", "dat", "mp3", "mp4", "wav",
        "ogg", "flac", "ttf", "woff", "woff2", "eot", "pyc", "class",
    ];
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        binary_exts.contains(&ext.to_lowercase().as_str())
    } else {
        false
    }
}
