//! Compaction engine for the LSM-like segment hierarchy.
//!
//! # Overview
//!
//! The [`CompactionEngine`] sits on top of [`SegmentManager`] and decides
//! *when* and *which* segments should be merged, using the three-level tier
//! model:
//!
//! | Level | Trigger                         | Action                           |
//! |-------|---------------------------------|----------------------------------|
//! | L1→L2 | L1 count ≥ `l1_threshold` (32)  | Compact oldest `l1_merge_batch` L1s into one L2 |
//! | L2→L2 | L2 count ≥ `l2_threshold` (4)   | Compact two oldest L2s into one larger L2 |
//! | L2→L3 | L2 total size > L3 × ratio (20%)| Compact all L2s into L3          |
//! | Rebuild| segment deletion ratio > 30%   | Re-compact the offending segment |
//!
//! # Usage
//!
//! ```ignore
//! let engine = CompactionEngine::default();
//! if let Some(stats) = engine.compact_if_needed(&mut segment_mgr, &load_content)? {
//!     println!("compacted {} segments", stats.merged_segments);
//! }
//! ```

use std::path::Path;

use anyhow::Result;

use crate::frequency_table::default_weight_fn;
use crate::index::IndexBuilder;
use crate::segment::{SegmentLevel, SegmentManager};

// ---------------------------------------------------------------------------
// CompactionConfig
// ---------------------------------------------------------------------------

/// Tuning knobs for the compaction engine.
///
/// Default values are conservative and work well for most repositories.
/// Adjust after benchmarking on your target workload.
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Number of L1 segments that triggers an L1→L2 compaction.
    pub l1_threshold: usize,
    /// Number of L2 segments that triggers an L2→L2 merge.
    pub l2_threshold: usize,
    /// Ratio of (L2 total size) / (L3 total size) above which all L2s are
    /// merged into L3.
    pub l2_to_l3_ratio: f64,
    /// Deletion ratio above which a single segment is rebuilt in place.
    pub deletion_rebuild_ratio: f64,
    /// Maximum number of L1 segments merged in a single L1→L2 compaction pass.
    pub l1_merge_batch: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            l1_threshold: 32,
            l2_threshold: 4,
            l2_to_l3_ratio: 0.2,
            deletion_rebuild_ratio: 0.3,
            l1_merge_batch: 16,
        }
    }
}

// ---------------------------------------------------------------------------
// CompactionPlan
// ---------------------------------------------------------------------------

/// Describes a single compaction operation to be performed.
#[derive(Debug, Clone)]
pub struct CompactionPlan {
    /// IDs of segments that will be merged.
    pub source_ids: Vec<u64>,
    /// Level of the resulting merged segment.
    pub target_level: SegmentLevel,
}

// ---------------------------------------------------------------------------
// CompactionStats
// ---------------------------------------------------------------------------

/// Statistics returned after a compaction pass.
#[derive(Debug, Default, Clone)]
pub struct CompactionStats {
    /// Number of source segments that were merged / removed.
    pub merged_segments: usize,
    /// Total live-file count across all source segments before compaction.
    pub files_before: usize,
    /// Live-file count in the resulting merged segment.
    pub files_after: usize,
    /// Approximate bytes reclaimed from deleted/tombstoned entries.
    pub bytes_reclaimed: u64,
}

// ---------------------------------------------------------------------------
// CompactionEngine
// ---------------------------------------------------------------------------

/// Decides when and what to compact, then delegates to [`SegmentManager`].
#[derive(Default)]
pub struct CompactionEngine {
    config: CompactionConfig,
}

impl CompactionEngine {
    /// Create an engine with the given configuration.
    pub fn new(config: CompactionConfig) -> Self {
        Self { config }
    }

    /// Return the active configuration.
    pub fn config(&self) -> &CompactionConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // Planning
    // -----------------------------------------------------------------------

    /// Inspect `mgr` and return the highest-priority compaction plan, or
    /// `None` if no compaction is needed right now.
    ///
    /// The precedence order is:
    /// 1. Single-segment rebuild (deletion ratio > threshold)
    /// 2. L1 → L2 batch merge
    /// 3. L2 → L2 pair merge
    /// 4. L2 → L3 full promotion
    pub fn should_compact(&self, mgr: &SegmentManager) -> Option<CompactionPlan> {
        // ── Priority 1: single-segment rebuild ────────────────────────────
        for seg in mgr.segments() {
            if let Some(stats) = mgr.segment_stats(seg.id) {
                if stats.total_files > 0
                    && stats.deletion_ratio > self.config.deletion_rebuild_ratio
                {
                    return Some(CompactionPlan {
                        source_ids: vec![seg.id],
                        target_level: seg.level,
                    });
                }
            }
        }

        // ── Priority 2: L1 → L2 batch ─────────────────────────────────────
        let l1_segs = mgr.segments_at_level(SegmentLevel::L1);
        if l1_segs.len() >= self.config.l1_threshold {
            let batch: Vec<u64> = l1_segs
                .iter()
                .take(self.config.l1_merge_batch)
                .map(|s| s.id)
                .collect();
            return Some(CompactionPlan {
                source_ids: batch,
                target_level: SegmentLevel::L2,
            });
        }

        // ── Priority 3: L2 pair merge ──────────────────────────────────────
        let l2_segs = mgr.segments_at_level(SegmentLevel::L2);
        if l2_segs.len() >= self.config.l2_threshold {
            let pair: Vec<u64> = l2_segs.iter().take(2).map(|s| s.id).collect();
            return Some(CompactionPlan {
                source_ids: pair,
                target_level: SegmentLevel::L2,
            });
        }

        // ── Priority 4: L2 → L3 promotion ─────────────────────────────────
        if !l2_segs.is_empty() {
            if let Some(l3_size) = self.total_level_disk_size(mgr, SegmentLevel::L3) {
                let l2_size = self
                    .total_level_disk_size(mgr, SegmentLevel::L2)
                    .unwrap_or(0);
                if l3_size > 0 && l2_size as f64 / l3_size as f64 > self.config.l2_to_l3_ratio {
                    // Merge all L2 segments plus existing L3 segments into a single L3.
                    // Exclude L1 (WAL) segments to preserve tiering.
                    let ids: Vec<u64> = mgr
                        .segments()
                        .iter()
                        .filter(|s| s.level == SegmentLevel::L2 || s.level == SegmentLevel::L3)
                        .map(|s| s.id)
                        .collect();
                    return Some(CompactionPlan {
                        source_ids: ids,
                        target_level: SegmentLevel::L3,
                    });
                }
            }
        }

        None
    }

    // -----------------------------------------------------------------------
    // Execution
    // -----------------------------------------------------------------------

    /// Execute a compaction plan.
    ///
    /// `load_content(path)` is called for every live file that needs to be
    /// re-indexed (only required when rebuilding a segment).  For posting-list
    /// merges the content is not re-read.
    ///
    /// Returns statistics about what was done, or an error if the operation
    /// fails.
    pub fn compact_segments<F>(
        &self,
        mgr: &mut SegmentManager,
        plan: &CompactionPlan,
        load_content: &F,
    ) -> Result<CompactionStats>
    where
        F: Fn(&Path) -> Option<Vec<u8>>,
    {
        let source_ids = &plan.source_ids;

        // Count files before compaction.
        let files_before: usize = source_ids
            .iter()
            .filter_map(|&id| mgr.segment_stats(id))
            .map(|s| s.total_files.saturating_sub(s.deleted_files))
            .sum();

        // Measure disk before.
        let bytes_before: u64 = source_ids
            .iter()
            .filter_map(|&id| mgr.segment_disk_size(id).ok())
            .sum();

        let new_id = if source_ids.len() == 1 {
            // Single-segment rebuild: re-index from content.  Returns `None`
            // when some live files could not be loaded — the rebuild is
            // skipped and the old segment kept, so unreadable files are never
            // silently dropped from the index.
            match self.rebuild_segment_inner(mgr, source_ids[0], plan.target_level, load_content)? {
                Some(id) => id,
                None => {
                    return Ok(CompactionStats {
                        merged_segments: 0,
                        files_before,
                        files_after: files_before,
                        bytes_reclaimed: 0,
                    });
                }
            }
        } else {
            // Multi-segment posting-list merge (no content needed).
            mgr.compact_segments(source_ids, plan.target_level)?
        };

        let files_after = mgr
            .segment_stats(new_id)
            .map(|s| s.total_files.saturating_sub(s.deleted_files))
            .unwrap_or(0);

        let bytes_after = mgr.segment_disk_size(new_id).unwrap_or(0);
        let bytes_reclaimed = bytes_before.saturating_sub(bytes_after);

        Ok(CompactionStats {
            merged_segments: source_ids.len(),
            files_before,
            files_after,
            bytes_reclaimed,
        })
    }

    /// Compact if the current state of `mgr` warrants it.
    ///
    /// Returns `Some(stats)` when a compaction was performed, `None` when
    /// nothing needed to be done.
    pub fn compact_if_needed<F>(
        &self,
        mgr: &mut SegmentManager,
        load_content: &F,
    ) -> Result<Option<CompactionStats>>
    where
        F: Fn(&Path) -> Option<Vec<u8>>,
    {
        match self.should_compact(mgr) {
            None => Ok(None),
            Some(plan) => {
                let stats = self.compact_segments(mgr, &plan, load_content)?;
                Ok(Some(stats))
            }
        }
    }

    /// Rebuild a single segment by re-indexing its live files.
    ///
    /// This reads file content via `load_content`, which allows stale
    /// n-gram data from deleted/updated files to be purged entirely.
    ///
    /// Returns the id of the new replacement segment.
    pub fn rebuild_segment<F>(
        &self,
        mgr: &mut SegmentManager,
        segment_id: u64,
        load_content: &F,
    ) -> Result<u64>
    where
        F: Fn(&Path) -> Option<Vec<u8>>,
    {
        let level = mgr
            .segments()
            .iter()
            .find(|s| s.id == segment_id)
            .map(|s| s.level)
            .ok_or_else(|| anyhow::anyhow!("rebuild_segment: segment {segment_id} not found"))?;
        match self.rebuild_segment_inner(mgr, segment_id, level, load_content)? {
            Some(new_id) => Ok(new_id),
            None => anyhow::bail!(
                "rebuild_segment: some live files of segment {segment_id} could not be loaded; \
                 the segment was left unchanged"
            ),
        }
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    /// Re-index a segment's live files and replace it with the new segment.
    ///
    /// Returns `Ok(Some(new_id))` on success, or `Ok(None)` when at least one
    /// live file could not be loaded — in that case the old segment is left
    /// untouched so the files remain searchable.
    fn rebuild_segment_inner<F>(
        &self,
        mgr: &mut SegmentManager,
        segment_id: u64,
        target_level: SegmentLevel,
        load_content: &F,
    ) -> Result<Option<u64>>
    where
        F: Fn(&Path) -> Option<Vec<u8>>,
    {
        // Collect live (non-deleted) paths from the segment.
        let live_paths: Vec<std::path::PathBuf> = {
            let seg = mgr
                .segments()
                .iter()
                .find(|s| s.id == segment_id)
                .ok_or_else(|| {
                    anyhow::anyhow!("rebuild_segment: segment {segment_id} not found")
                })?;
            seg.index
                .reader
                .file_paths
                .iter()
                .enumerate()
                .filter(|(fid, path)| {
                    !path.as_os_str().is_empty()
                        && !mgr.deletions_ref().is_deleted(segment_id, *fid as u32)
                })
                .map(|(_, path)| path.clone())
                .collect()
        };

        // Re-index from content.  Load everything up front: if any live file
        // fails to load we must not proceed, because the replacement segment
        // would permanently drop it from the index.
        let weight_fn = default_weight_fn();
        let mut builder = IndexBuilder::new(weight_fn);
        let mut load_failures = 0usize;
        for path in &live_paths {
            match load_content(path) {
                Some(content) => builder.add_file(path, &content)?,
                None => load_failures += 1,
            }
        }
        if load_failures > 0 {
            // Skip the rebuild; the caller retries on a later compaction pass.
            return Ok(None);
        }

        let new_id = mgr.add_segment(builder, None, target_level)?;
        mgr.remove_segment(segment_id)?;
        Ok(Some(new_id))
    }

    /// Sum of on-disk byte sizes for all segments at `level`.
    fn total_level_disk_size(&self, mgr: &SegmentManager, level: SegmentLevel) -> Option<u64> {
        let total: u64 = mgr
            .segments_at_level(level)
            .iter()
            .filter_map(|s| mgr.segment_disk_size(s.id).ok())
            .sum();
        Some(total)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::IndexBuilder;
    use crate::segment::SegmentManager;
    use std::path::Path;
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
    ) -> impl Fn(&Path) -> Option<Vec<u8>> + 'a {
        move |path: &Path| {
            files
                .iter()
                .find(|(p, _)| Path::new(p) == path)
                .map(|(_, c)| c.to_vec())
        }
    }

    // -----------------------------------------------------------------------
    // CompactionConfig defaults
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_config_values() {
        let config = CompactionConfig::default();
        assert_eq!(config.l1_threshold, 32);
        assert_eq!(config.l2_threshold, 4);
        assert_eq!(config.l2_to_l3_ratio, 0.2);
        assert_eq!(config.deletion_rebuild_ratio, 0.3);
        assert_eq!(config.l1_merge_batch, 16);
    }

    // -----------------------------------------------------------------------
    // should_compact planning
    // -----------------------------------------------------------------------

    #[test]
    fn test_should_compact_no_action_when_below_thresholds() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        // Add some L1 segments but below threshold (32)
        for i in 0..5 {
            mgr.add_segment(
                make_segment_builder(&[(format!("file{}.rs", i).as_str(), b"fn test() {}")])?,
                None,
                SegmentLevel::L1,
            )?;
        }

        let engine = CompactionEngine::default();
        let plan = engine.should_compact(&mgr);
        assert!(plan.is_none(), "no compaction should be needed");
        Ok(())
    }

    #[test]
    fn test_should_compact_l1_to_l2_when_threshold_reached() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        // Add exactly l1_threshold (32) segments
        for i in 0..32 {
            mgr.add_segment(
                make_segment_builder(&[(format!("file{}.rs", i).as_str(), b"fn test() {}")])?,
                None,
                SegmentLevel::L1,
            )?;
        }

        let engine = CompactionEngine::default();
        let plan = engine.should_compact(&mgr).expect("should have a plan");

        assert_eq!(plan.target_level, SegmentLevel::L2);
        // Should compact l1_merge_batch (16) segments
        assert_eq!(plan.source_ids.len(), 16);
        Ok(())
    }

    #[test]
    fn test_should_compact_l2_pair_merge_when_threshold_reached() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        // Add exactly l2_threshold (4) L2 segments
        for i in 0..4 {
            mgr.add_segment(
                make_segment_builder(&[(format!("file{}.rs", i).as_str(), b"fn test() {}")])?,
                None,
                SegmentLevel::L2,
            )?;
        }

        let engine = CompactionEngine::default();
        let plan = engine.should_compact(&mgr).expect("should have a plan");

        assert_eq!(plan.target_level, SegmentLevel::L2);
        // Should compact 2 oldest L2 segments
        assert_eq!(plan.source_ids.len(), 2);
        Ok(())
    }

    #[test]
    fn test_should_compact_single_segment_rebuild_on_high_deletion() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        // Create a segment with multiple files
        let files: &[(&str, &[u8])] = &[
            ("a.rs", b"fn a() {}"),
            ("b.rs", b"fn b() {}"),
            ("c.rs", b"fn c() {}"),
            ("d.rs", b"fn d() {}"),
        ];
        let id = mgr.add_segment(make_segment_builder(files)?, None, SegmentLevel::L1)?;

        // Delete most files to exceed deletion_rebuild_ratio (0.3)
        mgr.mark_deleted(Path::new("a.rs"))?;
        mgr.mark_deleted(Path::new("b.rs"))?;
        mgr.mark_deleted(Path::new("c.rs"))?;

        let engine = CompactionEngine::default();
        let plan = engine
            .should_compact(&mgr)
            .expect("should have a rebuild plan");

        assert_eq!(plan.source_ids.len(), 1);
        assert_eq!(plan.source_ids[0], id);
        assert_eq!(plan.target_level, SegmentLevel::L1);
        Ok(())
    }

    #[test]
    fn test_should_compact_rebuild_has_highest_priority() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        // Create many L1 segments to trigger L1→L2
        for i in 0..32 {
            mgr.add_segment(
                make_segment_builder(&[(format!("file{}.rs", i).as_str(), b"fn test() {}")])?,
                None,
                SegmentLevel::L1,
            )?;
        }

        // Create a segment with high deletion ratio
        let files: &[(&str, &[u8])] = &[
            ("del_a.rs", b"fn a() {}"),
            ("del_b.rs", b"fn b() {}"),
            ("del_c.rs", b"fn c() {}"),
        ];
        let del_id = mgr.add_segment(make_segment_builder(files)?, None, SegmentLevel::L2)?;

        mgr.mark_deleted(Path::new("del_a.rs"))?;
        mgr.mark_deleted(Path::new("del_b.rs"))?;

        let engine = CompactionEngine::default();
        let plan = engine.should_compact(&mgr).expect("should have a plan");

        // Rebuild should take priority over L1→L2
        assert_eq!(plan.source_ids.len(), 1);
        assert_eq!(plan.source_ids[0], del_id);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // compact_segments execution
    // -----------------------------------------------------------------------

    #[test]
    fn test_compact_l1_batch_to_l2() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        let files: &[(&str, &[u8])] = &[
            ("a.rs", b"fn alpha() {}"),
            ("b.rs", b"fn beta() {}"),
            ("c.rs", b"fn gamma() {}"),
        ];

        // Create 3 L1 segments
        let id0 = mgr.add_segment(make_segment_builder(&[files[0]])?, None, SegmentLevel::L1)?;
        let id1 = mgr.add_segment(make_segment_builder(&[files[1]])?, None, SegmentLevel::L1)?;
        let id2 = mgr.add_segment(make_segment_builder(&[files[2]])?, None, SegmentLevel::L1)?;

        let engine = CompactionEngine::default();
        let plan = CompactionPlan {
            source_ids: vec![id0, id1, id2],
            target_level: SegmentLevel::L2,
        };

        let stats = engine.compact_segments(&mut mgr, &plan, &content_loader(files))?;

        // Verify compaction happened
        assert_eq!(stats.merged_segments, 3);
        assert_eq!(mgr.l1_count(), 0);
        assert_eq!(mgr.l2_count(), 1);

        // Verify all files are searchable
        let r_alpha = mgr.search("alpha", content_loader(files), None)?;
        let r_beta = mgr.search("beta", content_loader(files), None)?;
        let r_gamma = mgr.search("gamma", content_loader(files), None)?;
        assert_eq!(r_alpha.len(), 1);
        assert_eq!(r_beta.len(), 1);
        assert_eq!(r_gamma.len(), 1);

        Ok(())
    }

    #[test]
    fn test_compact_l2_pair_merge() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        let files: &[(&str, &[u8])] = &[("a.rs", b"fn alpha() {}"), ("b.rs", b"fn beta() {}")];

        // Create 2 L2 segments
        let id0 = mgr.add_segment(make_segment_builder(&[files[0]])?, None, SegmentLevel::L2)?;
        let id1 = mgr.add_segment(make_segment_builder(&[files[1]])?, None, SegmentLevel::L2)?;

        let engine = CompactionEngine::default();
        let plan = CompactionPlan {
            source_ids: vec![id0, id1],
            target_level: SegmentLevel::L2,
        };

        let stats = engine.compact_segments(&mut mgr, &plan, &content_loader(files))?;

        // Verify merge happened
        assert_eq!(stats.merged_segments, 2);
        assert_eq!(mgr.l2_count(), 1);

        // Verify searchability
        let results = mgr.search("fn", content_loader(files), None)?;
        assert_eq!(results.len(), 2);

        Ok(())
    }

    #[test]
    fn test_compact_l2_to_l3_promotion() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        let files: &[(&str, &[u8])] = &[("a.rs", b"fn alpha() {}"), ("b.rs", b"fn beta() {}")];

        // Create 2 L2 segments
        let id0 = mgr.add_segment(make_segment_builder(&[files[0]])?, None, SegmentLevel::L2)?;
        let id1 = mgr.add_segment(make_segment_builder(&[files[1]])?, None, SegmentLevel::L2)?;

        let engine = CompactionEngine::default();
        let plan = CompactionPlan {
            source_ids: vec![id0, id1],
            target_level: SegmentLevel::L3,
        };

        engine.compact_segments(&mut mgr, &plan, &content_loader(files))?;

        // Verify promotion to L3
        assert_eq!(mgr.l2_count(), 0);
        assert_eq!(mgr.segments_at_level(SegmentLevel::L3).len(), 1);

        Ok(())
    }

    // -----------------------------------------------------------------------
    // compact_if_needed end-to-end
    // -----------------------------------------------------------------------

    #[test]
    fn test_compact_if_needed_triggers_l1_to_l2() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        // Add enough L1 segments to trigger compaction
        for i in 0..32 {
            mgr.add_segment(
                make_segment_builder(&[(format!("file{}.rs", i).as_str(), b"fn test() {}")])?,
                None,
                SegmentLevel::L1,
            )?;
        }

        let files: Vec<(String, &[u8])> = (0..32)
            .map(|i| (format!("file{}.rs", i), b"fn test() {}".as_ref()))
            .collect();
        let file_refs: Vec<(&str, &[u8])> = files.iter().map(|(s, b)| (s.as_str(), *b)).collect();

        let engine = CompactionEngine::default();
        let stats = engine
            .compact_if_needed(&mut mgr, &content_loader(&file_refs))?
            .expect("should trigger compaction");

        assert_eq!(stats.merged_segments, 16); // l1_merge_batch
        assert!(mgr.l1_count() < 32);
        assert!(mgr.l2_count() > 0);

        Ok(())
    }

    #[test]
    fn test_compact_if_needed_returns_none_when_not_needed() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        // Add only a few L1 segments (below threshold)
        for i in 0..5 {
            mgr.add_segment(
                make_segment_builder(&[(format!("file{}.rs", i).as_str(), b"fn test() {}")])?,
                None,
                SegmentLevel::L1,
            )?;
        }

        let files: Vec<(String, &[u8])> = (0..5)
            .map(|i| (format!("file{}.rs", i), b"fn test() {}".as_ref()))
            .collect();
        let file_refs: Vec<(&str, &[u8])> = files.iter().map(|(s, b)| (s.as_str(), *b)).collect();

        let engine = CompactionEngine::default();
        let result = engine.compact_if_needed(&mut mgr, &content_loader(&file_refs))?;

        assert!(result.is_none(), "should not compact when below threshold");
        Ok(())
    }

    // -----------------------------------------------------------------------
    // rebuild_segment
    // -----------------------------------------------------------------------

    #[test]
    fn test_rebuild_segment_purges_deleted_files() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        let files: &[(&str, &[u8])] = &[
            ("keep.rs", b"fn keep() {}"),
            ("delete.rs", b"fn delete() {}"),
        ];

        let id = mgr.add_segment(make_segment_builder(files)?, None, SegmentLevel::L1)?;

        // Mark one file as deleted
        mgr.mark_deleted(Path::new("delete.rs"))?;

        let engine = CompactionEngine::default();
        let new_id = engine.rebuild_segment(&mut mgr, id, &content_loader(files))?;

        // Verify old segment is gone, new segment exists
        assert!(mgr.segments().iter().all(|s| s.id != id));
        assert!(mgr.segments().iter().any(|s| s.id == new_id));

        // Verify deleted file is not searchable
        let results = mgr.search("delete", content_loader(files), None)?;
        assert!(results.is_empty(), "deleted file should be purged");

        // Verify live file is still searchable
        let results = mgr.search("keep", content_loader(files), None)?;
        assert_eq!(results.len(), 1);

        Ok(())
    }

    #[test]
    fn test_rebuild_segment_preserves_live_files() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        let files: &[(&str, &[u8])] = &[
            ("a.rs", b"fn alpha() {}"),
            ("b.rs", b"fn beta() {}"),
            ("c.rs", b"fn gamma() {}"),
        ];

        let id = mgr.add_segment(make_segment_builder(files)?, None, SegmentLevel::L2)?;

        // Delete one file
        mgr.mark_deleted(Path::new("b.rs"))?;

        let engine = CompactionEngine::default();
        engine.rebuild_segment(&mut mgr, id, &content_loader(files))?;

        // Verify live files are searchable
        let r_alpha = mgr.search("alpha", content_loader(files), None)?;
        let r_gamma = mgr.search("gamma", content_loader(files), None)?;
        assert_eq!(r_alpha.len(), 1);
        assert_eq!(r_gamma.len(), 1);

        // Verify deleted file is not searchable
        let r_beta = mgr.search("beta", content_loader(files), None)?;
        assert!(r_beta.is_empty());

        Ok(())
    }

    #[test]
    fn test_rebuild_skipped_when_live_file_unloadable() -> Result<()> {
        // FN-2 regression: if a live file cannot be loaded during a
        // single-segment rebuild, the rebuild must be skipped and the old
        // segment kept — otherwise the file silently disappears from the
        // index.
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        let files: &[(&str, &[u8])] = &[
            ("gone.rs", b"fn gone() {}"),
            ("keep.rs", b"fn keep_unloadable_marker() {}"),
        ];
        let id = mgr.add_segment(make_segment_builder(files)?, None, SegmentLevel::L2)?;

        // Push the deletion ratio over the threshold to trigger a rebuild.
        mgr.mark_deleted(Path::new("gone.rs"))?;

        let engine = CompactionEngine::default();
        let plan = engine.should_compact(&mgr).expect("rebuild plan expected");
        assert_eq!(plan.source_ids, vec![id]);

        // Loader that cannot read keep.rs.
        let failing_loader = |path: &Path| -> Option<Vec<u8>> {
            if path == Path::new("keep.rs") {
                None
            } else {
                Some(b"fn gone() {}".to_vec())
            }
        };
        let stats = engine.compact_segments(&mut mgr, &plan, &failing_loader)?;
        assert_eq!(stats.merged_segments, 0, "rebuild must be skipped");
        assert!(
            mgr.segments().iter().any(|s| s.id == id),
            "old segment must be preserved"
        );

        // With a working loader the file is still searchable afterwards.
        let results = mgr.search("keep_unloadable_marker", content_loader(files), None)?;
        assert_eq!(results.len(), 1, "unloadable file must not be dropped");

        // And a later pass with a working loader performs the rebuild.
        let stats = engine.compact_segments(&mut mgr, &plan, &content_loader(files))?;
        assert_eq!(stats.merged_segments, 1);
        let results = mgr.search("keep_unloadable_marker", content_loader(files), None)?;
        assert_eq!(results.len(), 1);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // CompactionStats verification
    // -----------------------------------------------------------------------

    #[test]
    fn test_compaction_stats_reports_correct_counts() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut mgr = SegmentManager::create(tmp.path())?;

        let files: &[(&str, &[u8])] = &[
            ("a.rs", b"fn alpha() {}"),
            ("b.rs", b"fn beta() {}"),
            ("c.rs", b"fn gamma() {}"),
        ];

        // Create 3 segments
        let id0 = mgr.add_segment(make_segment_builder(&[files[0]])?, None, SegmentLevel::L1)?;
        let id1 = mgr.add_segment(make_segment_builder(&[files[1]])?, None, SegmentLevel::L1)?;
        let id2 = mgr.add_segment(make_segment_builder(&[files[2]])?, None, SegmentLevel::L1)?;

        let engine = CompactionEngine::default();
        let plan = CompactionPlan {
            source_ids: vec![id0, id1, id2],
            target_level: SegmentLevel::L2,
        };

        let stats = engine.compact_segments(&mut mgr, &plan, &content_loader(files))?;

        // Verify stats
        assert_eq!(stats.merged_segments, 3);
        assert_eq!(stats.files_before, 3);
        assert_eq!(stats.files_after, 3);

        Ok(())
    }
}
