//! Per-segment deletion bitmaps.
//!
//! `postings.bin` is immutable once written, so deleted files cannot be
//! removed in place.  Instead each segment carries an external bitset that
//! marks which of its `file_id`s should be treated as deleted during search.
//!
//! # Binary format (`deletion.bin`)
//!
//! ```text
//! [0..4]  word_count: u32 LE
//! [4..]   word_count × u64 LE words  (bit i → file_id i is deleted)
//! ```

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Result};

// ---------------------------------------------------------------------------
// DeletionBitmap
// ---------------------------------------------------------------------------

/// A compact bitset that records which `file_id`s are deleted within one segment.
#[derive(Debug, Default, Clone)]
pub struct DeletionBitmap {
    words: Vec<u64>,
}

impl DeletionBitmap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark `file_id` as deleted.
    pub fn mark(&mut self, file_id: u32) {
        let idx = (file_id / 64) as usize;
        let bit = file_id % 64;
        if idx >= self.words.len() {
            self.words.resize(idx + 1, 0);
        }
        self.words[idx] |= 1u64 << bit;
    }

    /// Return `true` if `file_id` is deleted.
    pub fn is_deleted(&self, file_id: u32) -> bool {
        let idx = (file_id / 64) as usize;
        let bit = file_id % 64;
        idx < self.words.len() && (self.words[idx] >> bit) & 1 == 1
    }

    /// Return `true` if no file_id has been marked deleted.
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(4 + self.words.len() * 8);
        let count = self.words.len() as u32;
        buf.extend_from_slice(&count.to_le_bytes());
        for &w in &self.words {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        buf
    }

    /// Deserialize from bytes produced by [`Self::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 4 {
            bail!("DeletionBitmap: too short ({} bytes)", bytes.len());
        }
        let count = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
        let expected = 4 + count * 8;
        if bytes.len() < expected {
            bail!(
                "DeletionBitmap: truncated (have {} bytes, need {})",
                bytes.len(),
                expected
            );
        }
        let mut words = Vec::with_capacity(count);
        for i in 0..count {
            let off = 4 + i * 8;
            words.push(u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()));
        }
        Ok(Self { words })
    }
}

// ---------------------------------------------------------------------------
// SegmentDeletions
// ---------------------------------------------------------------------------

/// Deletion bitmaps for all segments in a [`super::segment::SegmentManager`],
/// keyed by segment id.
#[derive(Debug, Default)]
pub struct SegmentDeletions {
    bitmaps: HashMap<u64, DeletionBitmap>,
}

impl SegmentDeletions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark `file_id` as deleted in `segment_id`.
    pub fn mark(&mut self, segment_id: u64, file_id: u32) {
        self.bitmaps.entry(segment_id).or_default().mark(file_id);
    }

    /// Return `true` if `file_id` is deleted in `segment_id`.
    pub fn is_deleted(&self, segment_id: u64, file_id: u32) -> bool {
        self.bitmaps
            .get(&segment_id)
            .is_some_and(|b| b.is_deleted(file_id))
    }

    /// Return the bitmap for `segment_id`, if one exists.
    pub fn bitmap_for(&self, segment_id: u64) -> Option<&DeletionBitmap> {
        self.bitmaps.get(&segment_id)
    }

    /// Remove the in-memory deletion bitmap for `segment_id`.
    ///
    /// Called when a segment is permanently deleted from the store so that
    /// memory for its bitmap is reclaimed.
    pub fn remove_segment(&mut self, segment_id: u64) {
        self.bitmaps.remove(&segment_id);
    }

    /// Load (or refresh) a segment's deletion bitmap from `segment_dir/deletion.bin`.
    ///
    /// If the file does not exist, any previously loaded bitmap for
    /// `segment_id` is removed so that the in-memory view stays consistent
    /// (e.g. after clearing deletions by deleting the file).
    pub fn load_segment(&mut self, segment_id: u64, segment_dir: &Path) -> Result<()> {
        let path = segment_dir.join("deletion.bin");
        if path.exists() {
            let bytes = std::fs::read(&path)?;
            let bitmap = DeletionBitmap::from_bytes(&bytes)?;
            self.bitmaps.insert(segment_id, bitmap);
        } else {
            self.bitmaps.remove(&segment_id);
        }
        Ok(())
    }

    /// Persist `segment_id`'s bitmap to `segment_dir/deletion.bin`.
    ///
    /// If there are no deletions for `segment_id`, any existing `deletion.bin`
    /// is removed so that [`Self::load_segment`] continues to work correctly
    /// (an empty or absent file would fail the 4-byte header check).
    pub fn save_segment(&self, segment_id: u64, segment_dir: &Path) -> Result<()> {
        let path = segment_dir.join("deletion.bin");
        match self.bitmaps.get(&segment_id) {
            Some(bitmap) if !bitmap.is_empty() => {
                // Atomic write: a crash mid-write must not leave a truncated
                // deletion.bin behind (that would make the whole store
                // unreadable on next open).
                crate::storage::write_atomic(&path, &bitmap.to_bytes())?;
            }
            _ => {
                // No deletions – remove the file so load_segment treats this
                // segment as having no deletions on the next open.
                if path.exists() {
                    std::fs::remove_file(&path)?;
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_mark_and_is_deleted() {
        let mut bm = DeletionBitmap::new();
        assert!(!bm.is_deleted(0));
        assert!(!bm.is_deleted(63));
        assert!(!bm.is_deleted(64));

        bm.mark(0);
        bm.mark(63);
        bm.mark(64);

        assert!(bm.is_deleted(0));
        assert!(bm.is_deleted(63));
        assert!(bm.is_deleted(64));
        assert!(!bm.is_deleted(1));
        assert!(!bm.is_deleted(65));
    }

    #[test]
    fn test_is_empty() {
        let mut bm = DeletionBitmap::new();
        assert!(bm.is_empty());
        bm.mark(5);
        assert!(!bm.is_empty());
    }

    #[test]
    fn test_roundtrip_bytes() {
        let mut bm = DeletionBitmap::new();
        bm.mark(0);
        bm.mark(127);
        bm.mark(200);

        let bytes = bm.to_bytes();
        let bm2 = DeletionBitmap::from_bytes(&bytes).unwrap();

        assert!(bm2.is_deleted(0));
        assert!(bm2.is_deleted(127));
        assert!(bm2.is_deleted(200));
        assert!(!bm2.is_deleted(1));
    }

    #[test]
    fn test_empty_roundtrip() {
        let bm = DeletionBitmap::new();
        let bytes = bm.to_bytes();
        let bm2 = DeletionBitmap::from_bytes(&bytes).unwrap();
        assert!(bm2.is_empty());
    }

    #[test]
    fn test_from_bytes_too_short() {
        assert!(DeletionBitmap::from_bytes(&[]).is_err());
        assert!(DeletionBitmap::from_bytes(&[1, 0, 0]).is_err());
    }

    #[test]
    fn test_from_bytes_truncated() {
        // 4-byte header says 2 words, but only 8 bytes follow (should be 16).
        let mut bytes = vec![2u8, 0, 0, 0];
        bytes.extend_from_slice(&0u64.to_le_bytes());
        assert!(DeletionBitmap::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_segment_deletions_mark_and_query() {
        let mut sd = SegmentDeletions::new();
        sd.mark(0, 5);
        sd.mark(1, 10);

        assert!(sd.is_deleted(0, 5));
        assert!(!sd.is_deleted(0, 10));
        assert!(sd.is_deleted(1, 10));
        assert!(!sd.is_deleted(1, 5));
        assert!(!sd.is_deleted(2, 5));
    }

    #[test]
    fn test_segment_deletions_load_save() -> Result<()> {
        let tmp = TempDir::new()?;
        let seg_dir = tmp.path();

        let mut sd = SegmentDeletions::new();
        sd.mark(0, 3);
        sd.mark(0, 7);
        sd.save_segment(0, seg_dir)?;

        let mut sd2 = SegmentDeletions::new();
        sd2.load_segment(0, seg_dir)?;

        assert!(sd2.is_deleted(0, 3));
        assert!(sd2.is_deleted(0, 7));
        assert!(!sd2.is_deleted(0, 0));

        Ok(())
    }

    #[test]
    fn test_load_missing_deletion_file_is_ok() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut sd = SegmentDeletions::new();
        // Should not error when the file doesn't exist.
        sd.load_segment(99, tmp.path())?;
        assert!(!sd.is_deleted(99, 0));
        Ok(())
    }

    #[test]
    fn test_load_segment_clears_stale_bitmap() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut sd = SegmentDeletions::new();
        // Pre-load a bitmap.
        sd.mark(0, 5);

        // load_segment with a missing file should clear the stale bitmap.
        sd.load_segment(0, tmp.path())?;
        assert!(
            !sd.is_deleted(0, 5),
            "stale bitmap should be cleared when file is absent"
        );
        Ok(())
    }

    #[test]
    fn test_save_segment_no_file_when_empty() -> Result<()> {
        let tmp = TempDir::new()?;
        let sd = SegmentDeletions::new(); // no entries for segment 0
        sd.save_segment(0, tmp.path())?;
        // deletion.bin must NOT be written when the bitmap is absent/empty.
        assert!(
            !tmp.path().join("deletion.bin").exists(),
            "deletion.bin should not exist when there are no deletions"
        );
        Ok(())
    }

    #[test]
    fn test_save_then_clear_removes_file() -> Result<()> {
        let tmp = TempDir::new()?;
        let mut sd = SegmentDeletions::new();
        sd.mark(0, 1);
        sd.save_segment(0, tmp.path())?;
        assert!(tmp.path().join("deletion.bin").exists());

        // Simulate clearing by removing the bitmap and re-saving.
        let sd2 = SegmentDeletions::new();
        sd2.save_segment(0, tmp.path())?;
        assert!(
            !tmp.path().join("deletion.bin").exists(),
            "deletion.bin should be removed when bitmap is cleared"
        );
        Ok(())
    }
}
