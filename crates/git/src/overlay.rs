//! In-memory overlay index for uncommitted working-tree changes.
//!
//! The overlay stores a mini inverted index for files that have been modified
//! since the base commit but not yet committed.  During a search, the overlay
//! results are merged with the base index results: overlay entries shadow
//! base-index entries for the same file path.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::Result;
use sparse_ngram_core::default_weight_fn;
use sparse_ngram_core::ngram::{build_all_hashes, WeightFn};

// ---------------------------------------------------------------------------
// OverlayIndex
// ---------------------------------------------------------------------------

/// An in-memory n-gram index for modified-but-uncommitted files.
#[derive(Default)]
pub struct OverlayIndex {
    /// file path → set of n-gram hashes present in that file.
    file_ngrams: HashMap<PathBuf, HashSet<u64>>,
    /// n-gram hash → set of file paths.
    hash_to_paths: BTreeMap<u64, HashSet<PathBuf>>,
    /// Paths that have been deleted (and should be hidden from base results).
    deleted: HashSet<PathBuf>,
    weight_fn: Option<Box<dyn WeightFn>>,
}

impl OverlayIndex {
    pub fn new() -> Self {
        Self {
            file_ngrams: HashMap::new(),
            hash_to_paths: BTreeMap::new(),
            deleted: HashSet::new(),
            weight_fn: Some(default_weight_fn()),
        }
    }

    fn wf(&self) -> &dyn WeightFn {
        self.weight_fn.as_deref().unwrap()
    }

    // -----------------------------------------------------------------------
    // Mutations
    // -----------------------------------------------------------------------

    /// Index (or re-index) a modified file.
    pub fn add_file(&mut self, path: &Path, content: &[u8]) -> Result<()> {
        // Remove any previous entry for this path.
        self.remove_internal(path);
        self.deleted.remove(path);

        let hashes_vec = build_all_hashes(content, self.wf());
        let mut hashes = HashSet::with_capacity(hashes_vec.len());

        for h in hashes_vec {
            hashes.insert(h);
            self.hash_to_paths
                .entry(h)
                .or_default()
                .insert(path.to_path_buf());
        }

        self.file_ngrams.insert(path.to_path_buf(), hashes);
        Ok(())
    }

    /// Mark `path` as deleted (removed from the working tree).
    pub fn remove_file(&mut self, path: &Path) {
        self.remove_internal(path);
        self.deleted.insert(path.to_path_buf());
    }

    /// Remove all overlay data and clear the deleted set.
    pub fn clear(&mut self) {
        self.file_ngrams.clear();
        self.hash_to_paths.clear();
        self.deleted.clear();
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Return paths in the overlay that contain *all* of the given n-gram hashes.
    pub fn lookup_and(&self, hashes: &[u64]) -> HashSet<PathBuf> {
        if hashes.is_empty() {
            return self.file_ngrams.keys().cloned().collect();
        }
        // Start with the set for the first hash.
        let mut candidates: HashSet<PathBuf> = match self.hash_to_paths.get(&hashes[0]) {
            None => return HashSet::new(),
            Some(s) => s.clone(),
        };
        for &h in &hashes[1..] {
            match self.hash_to_paths.get(&h) {
                None => return HashSet::new(),
                Some(s) => candidates.retain(|p| s.contains(p)),
            }
            if candidates.is_empty() {
                break;
            }
        }
        candidates
    }

    /// Return paths in the overlay that contain *any* of the given n-gram hash sets.
    pub fn lookup_or(&self, hash_sets: &[Vec<u64>]) -> HashSet<PathBuf> {
        let mut result = HashSet::new();
        for hashes in hash_sets {
            result.extend(self.lookup_and(hashes));
        }
        result
    }

    /// Returns true if `path` is in the deleted set.
    pub fn is_deleted(&self, path: &Path) -> bool {
        self.deleted.contains(path)
    }

    /// Returns true if `path` has an overlay entry.
    pub fn is_modified(&self, path: &Path) -> bool {
        self.file_ngrams.contains_key(path)
    }

    /// Paths that are modified (have overlay entries).
    pub fn modified_paths(&self) -> impl Iterator<Item = &Path> {
        self.file_ngrams.keys().map(PathBuf::as_path)
    }

    /// Paths that are deleted.
    pub fn deleted_paths(&self) -> impl Iterator<Item = &Path> {
        self.deleted.iter().map(PathBuf::as_path)
    }

    pub fn is_empty(&self) -> bool {
        self.file_ngrams.is_empty() && self.deleted.is_empty()
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    fn remove_internal(&mut self, path: &Path) {
        if let Some(hashes) = self.file_ngrams.remove(path) {
            for h in hashes {
                if let Some(set) = self.hash_to_paths.get_mut(&h) {
                    set.remove(path);
                    if set.is_empty() {
                        self.hash_to_paths.remove(&h);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_lookup() -> Result<()> {
        let mut overlay = OverlayIndex::new();
        overlay.add_file(Path::new("a.rs"), b"fn main() {}")?;
        overlay.add_file(Path::new("b.rs"), b"fn helper() {}")?;

        // Both files share "fn " — look up any n-gram hash via build_all.
        use sparse_ngram_core::{build_all, default_weight_fn};
        let wf = default_weight_fn();
        let ngrams = build_all(b"fn", wf.as_ref());
        if let Some(ng) = ngrams.first() {
            let h = ng.hash();
            let result = overlay.lookup_and(&[h]);
            let _ = result;
        }
        Ok(())
    }

    #[test]
    fn test_remove_file() -> Result<()> {
        let mut overlay = OverlayIndex::new();
        overlay.add_file(Path::new("a.rs"), b"fn main() {}")?;
        overlay.remove_file(Path::new("a.rs"));

        assert!(overlay.is_deleted(Path::new("a.rs")));
        assert!(!overlay.is_modified(Path::new("a.rs")));
        Ok(())
    }

    #[test]
    fn test_clear() -> Result<()> {
        let mut overlay = OverlayIndex::new();
        overlay.add_file(Path::new("a.rs"), b"fn main() {}")?;
        overlay.remove_file(Path::new("b.rs"));
        overlay.clear();
        assert!(overlay.is_empty());
        Ok(())
    }

    #[test]
    fn test_re_index_replaces_old() -> Result<()> {
        let mut overlay = OverlayIndex::new();
        overlay.add_file(Path::new("a.rs"), b"fn main() {}")?;
        // Re-index with different content.
        overlay.add_file(Path::new("a.rs"), b"struct Foo {}")?;
        // Still only one entry for the path.
        assert_eq!(overlay.file_ngrams.len(), 1);
        Ok(())
    }
}
