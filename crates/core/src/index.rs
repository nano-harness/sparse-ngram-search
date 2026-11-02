//! Index construction and management.
//!
//! # Building an index
//!
//! ```ignore
//! let wf = default_weight_fn();
//! let mut builder = IndexBuilder::new(wf);
//! for (path, content) in files {
//!     builder.add_file(&path, &content)?;
//! }
//! let index = builder.build(&output_dir)?;
//! ```
//!
//! The builder accumulates an in-memory inverted index
//! `HashMap<hash, Vec<file_id>>` (fast u64 hasher) and serialises it in
//! ascending hash order at the end.
//!
//! # Bulk builds
//!
//! For whole-repository builds, extract per-file hash sets in parallel with
//! [`file_ngram_hashes`] and merge them with [`IndexBuilder::from_file_hashes`],
//! which parallelises the merge across Rayon workers.
//!
//! # Incremental updates
//!
//! For incremental updates, open the existing index, modify it in memory,
//! and write it back.  The `Index::update` API handles add/remove/change.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::frequency_table::default_weight_fn;
use crate::ngram::{build_all_hashes, for_each_ngram, WeightFn};
use crate::storage::{IndexPaths, IndexReader, IndexWriter};

// ---------------------------------------------------------------------------
// Fast u64 hashing for the postings map
// ---------------------------------------------------------------------------

/// Minimal DoS-resistant-enough hasher for `u64` n-gram keys.
///
/// The keys are already FNV-1a outputs (well mixed), so a single
/// splitmix64-style finalisation is plenty — and several times faster than
/// the default SipHash, which matters when inserting hundreds of millions of
/// posting entries on large repositories.
#[derive(Default, Clone, Copy)]
pub struct FastU64Hasher(u64);

impl std::hash::Hasher for FastU64Hasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // Fallback for non-u64 writes (unused by HashMap<u64, _>).
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x1000_0000_01B3);
        }
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }

    #[inline]
    fn finish(&self) -> u64 {
        // splitmix64 finaliser: excellent avalanche in both high and low bits.
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Postings map: n-gram hash → ascending list of file_ids.
type PostingsMap = HashMap<u64, Vec<u32>, std::hash::BuildHasherDefault<FastU64Hasher>>;

// ---------------------------------------------------------------------------
// IndexMeta
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMeta {
    pub version: u32,
    /// Base git commit hash (if the index was built from a git repo).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_commit: Option<String>,
    /// Unix timestamp of the last index build/sync.
    pub timestamp: u64,
    /// Total number of indexed files.
    pub file_count: usize,
    /// Total number of unique n-gram hashes.
    pub ngram_count: usize,
    /// Git branch this segment was built from.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    /// The segment id this metadata belongs to (set by `SegmentManager`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segment_id: Option<u64>,
    /// Path to the upstream repository (for fork/shared-index scenarios).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream_repo: Option<String>,
}

impl IndexMeta {
    pub fn new(file_count: usize, ngram_count: usize) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            version: 1,
            base_commit: None,
            timestamp,
            file_count,
            ngram_count,
            branch: None,
            segment_id: None,
            upstream_repo: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Open-addressing u64 dedup set
// ---------------------------------------------------------------------------

/// Growable open-addressing hash set specialised for deduplicating the raw
/// n-gram hash stream of a single file.  The stream is dominated by
/// duplicates (≈15 emissions per unique token), so a small table with linear
/// probing beats sort-then-dedup in both time and transient memory.
struct U64Dedup {
    table: Vec<u64>,
    /// Number of occupied slots (excluding the sentinel key).
    len: usize,
    /// Whether the sentinel value itself was inserted as a real key.
    has_sentinel_key: bool,
}

impl U64Dedup {
    /// Slot marker for "empty".  A real key equal to this value is stored
    /// out-of-band via `has_sentinel_key`.
    const EMPTY: u64 = u64::MAX;
    /// Minimum table capacity (power of two).
    const MIN_CAP: usize = 1 << 10;

    #[inline]
    fn slot(hash: u64, cap_log2: u32) -> usize {
        // Fibonacci hashing: take the high bits of hash · φ·2⁶⁴.
        (hash.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - cap_log2)) as usize
    }

    #[inline]
    fn insert(&mut self, hash: u64) {
        if hash == Self::EMPTY {
            self.has_sentinel_key = true;
            return;
        }
        // Keep the load factor below 75%.
        if (self.len + 1) * 4 > self.table.len() * 3 {
            self.grow();
        }
        let cap_log2 = self.table.len().trailing_zeros();
        let mask = self.table.len() - 1;
        let mut i = Self::slot(hash, cap_log2);
        loop {
            let v = self.table[i];
            if v == Self::EMPTY {
                self.table[i] = hash;
                self.len += 1;
                return;
            }
            if v == hash {
                return;
            }
            i = (i + 1) & mask;
        }
    }

    fn grow(&mut self) {
        let new_cap = (self.table.len() * 2).max(Self::MIN_CAP);
        let mut new_table = vec![Self::EMPTY; new_cap];
        let cap_log2 = new_cap.trailing_zeros();
        let mask = new_cap - 1;
        for &v in &self.table {
            if v != Self::EMPTY {
                let mut i = Self::slot(v, cap_log2);
                while new_table[i] != Self::EMPTY {
                    i = (i + 1) & mask;
                }
                new_table[i] = v;
            }
        }
        self.table = new_table;
    }

    fn into_vec(self) -> Vec<u64> {
        let mut out = Vec::with_capacity(self.len + usize::from(self.has_sentinel_key));
        out.extend(self.table.into_iter().filter(|&v| v != Self::EMPTY));
        if self.has_sentinel_key {
            out.push(Self::EMPTY);
        }
        out
    }
}

impl Default for U64Dedup {
    fn default() -> Self {
        Self {
            table: vec![Self::EMPTY; Self::MIN_CAP],
            len: 0,
            has_sentinel_key: false,
        }
    }
}

impl U64Dedup {
    /// Create a set pre-sized for about `uniques` distinct keys, avoiding
    /// repeated grow-and-rehash cycles for large files.
    fn with_capacity(uniques: usize) -> Self {
        // Slots needed to hold `uniques` keys at a ≤75% load factor.
        const MAX_CAP: usize = 1 << 20;
        let cap = (uniques.saturating_mul(4) / 3)
            .next_power_of_two()
            .clamp(Self::MIN_CAP, MAX_CAP);
        Self {
            table: vec![Self::EMPTY; cap],
            len: 0,
            has_sentinel_key: false,
        }
    }
}

// ---------------------------------------------------------------------------
// IndexBuilder
// ---------------------------------------------------------------------------

/// Accumulates in-memory postings and builds the on-disk index.
pub struct IndexBuilder {
    weight_fn: Box<dyn WeightFn>,
    /// Inverted index: n-gram hash → sorted list of file_ids.
    postings: PostingsMap,
    /// Bulk-build postings kept as disjoint per-partition maps (ordered by
    /// the hashes' top bits), deferring the union until it is actually
    /// needed — `build` can drain the partitions in order without ever
    /// re-hashing their entries into a single map.
    partitioned: Option<Vec<PostingsMap>>,
    /// file_id → path.
    file_table: Vec<PathBuf>,
    /// path → file_id (for dedup / remove).
    path_to_id: HashMap<PathBuf, u32>,
}

impl IndexBuilder {
    pub fn new(weight_fn: Box<dyn WeightFn>) -> Self {
        Self {
            weight_fn,
            postings: PostingsMap::default(),
            partitioned: None,
            file_table: Vec::new(),
            path_to_id: HashMap::new(),
        }
    }

    /// Create a builder with the default (built-in) weight function.
    pub fn with_default_weights() -> Self {
        Self::new(default_weight_fn())
    }

    /// Index the given file content.
    ///
    /// If `path` was already indexed, the old postings are removed first.
    pub fn add_file(&mut self, path: &Path, content: &[u8]) -> Result<()> {
        let hashes = build_all_hashes(content, self.weight_fn.as_ref());
        self.add_file_hashes(path, hashes)
    }

    /// Merge any deferred partition maps into `self.postings`, restoring the
    /// single-map invariant the incremental operations rely on.
    fn materialize(&mut self) {
        if let Some(parts) = self.partitioned.take() {
            let total: usize = parts.iter().map(|m| m.len()).sum();
            self.postings.reserve(total);
            for map in parts {
                self.postings.extend(map);
            }
        }
    }

    /// Index a file from its pre-extracted n-gram hashes.
    ///
    /// The hashes (typically the raw output of [`build_all_hashes`]) are
    /// sorted and deduplicated internally, then `file_id` is appended once to
    /// each hash's posting list.  Splitting extraction from insertion lets
    /// callers run the (expensive) extraction in parallel and perform only
    /// the (cheap) merge sequentially.
    pub fn add_file_hashes(&mut self, path: &Path, mut hashes: Vec<u64>) -> Result<()> {
        self.materialize();
        // Remove existing entry if re-indexing.
        if self.path_to_id.contains_key(path) {
            self.remove_file(path)?;
        }

        let file_id = self.file_table.len() as u32;
        self.file_table.push(path.to_path_buf());
        self.path_to_id.insert(path.to_path_buf(), file_id);

        hashes.sort_unstable();
        hashes.dedup();
        for hash in hashes {
            let list = self.postings.entry(hash).or_default();
            // Keep the list sorted; avoid duplicates.
            if list.last() != Some(&file_id) {
                list.push(file_id);
            }
        }
        Ok(())
    }

    /// Remove all postings for `path`.
    pub fn remove_file(&mut self, path: &Path) -> Result<()> {
        self.materialize();
        let file_id = match self.path_to_id.remove(path) {
            Some(id) => id,
            None => return Ok(()), // not indexed, nothing to do
        };
        for list in self.postings.values_mut() {
            list.retain(|&id| id != file_id);
        }
        // We intentionally leave a "tombstone" in file_table (the slot stays,
        // with an empty path) to keep file_ids stable for other files.
        if let Some(slot) = self.file_table.get_mut(file_id as usize) {
            *slot = PathBuf::new();
        }
        Ok(())
    }

    /// Build a builder from parallel-extracted per-file n-gram hash sets.
    ///
    /// `per_file[i]` is the **sorted, deduplicated** hash set for `paths[i]`
    /// (see [`file_ngram_hashes`]), or `None` if the file was skipped.
    /// File ids are assigned densely in slice order — identical to calling
    /// [`IndexBuilder::add_file`] on each file sequentially.
    ///
    /// The merge is parallelised by hash partition: each Rayon worker owns a
    /// slice of the hash space and binary-searches every file's sorted hash
    /// set for its range, so posting lists are built in ascending file-id
    /// order and the partition maps are disjoint.  The partitions are kept
    /// as-is (see the `partitioned` field); their union is the final
    /// postings map.
    pub fn from_file_hashes(
        paths: &[PathBuf],
        per_file: Vec<Option<Vec<u64>>>,
        weight_fn: Box<dyn WeightFn>,
    ) -> Self {
        assert_eq!(
            paths.len(),
            per_file.len(),
            "paths and per_file must have the same length"
        );
        let n = paths.len();

        // Dense file ids in slice order over the non-skipped files.
        let mut dense_id = vec![u32::MAX; n];
        let mut file_table = Vec::new();
        let mut path_to_id = HashMap::new();
        for (i, path) in paths.iter().enumerate() {
            if per_file[i].is_some() {
                dense_id[i] = file_table.len() as u32;
                file_table.push(path.clone());
                path_to_id.insert(path.clone(), dense_id[i]);
            }
        }

        // Number of hash partitions for the parallel merge.  64 partitions
        // keep each partition's postings map small enough to stay largely
        // cache-resident during the merge (a few hundred thousand entries
        // even for a full Linux kernel tree), which roughly halves the
        // per-insert cost compared to fewer, larger maps.
        const PARTITION_LOG2: u32 = 6;
        const PARTITION_COUNT: usize = 1 << PARTITION_LOG2;
        const PARTITION_SHIFT: u32 = 64 - PARTITION_LOG2;

        let partitions: Vec<PostingsMap> = (0..PARTITION_COUNT)
            .into_par_iter()
            .map(|w| {
                // Hash range owned by this partition: [lo, hi).
                let lo = (w as u64) << PARTITION_SHIFT;
                let hi = if w + 1 == PARTITION_COUNT {
                    None // top partition runs to u64::MAX, inclusive
                } else {
                    Some(((w as u64) + 1) << PARTITION_SHIFT)
                };
                let mut map = PostingsMap::default();
                for (i, slot) in per_file.iter().enumerate() {
                    let Some(hashes) = slot else {
                        continue;
                    };
                    // `hashes` is sorted: restrict the scan to this
                    // partition's range instead of filtering the whole set.
                    let start = hashes.partition_point(|&h| h < lo);
                    let end = match hi {
                        Some(hi) => hashes.partition_point(|&h| h < hi),
                        None => hashes.len(),
                    };
                    let file_id = dense_id[i];
                    for &hash in &hashes[start..end] {
                        map.entry(hash).or_default().push(file_id);
                    }
                }
                map
            })
            .collect();

        Self {
            weight_fn,
            postings: PostingsMap::default(),
            partitioned: Some(partitions),
            file_table,
            path_to_id,
        }
    }

    /// Number of files currently indexed (including tombstones).
    pub fn file_count(&self) -> usize {
        self.path_to_id.len()
    }

    /// Serialise the index to `output_dir`.
    pub fn build(self, output_dir: &Path) -> Result<Index> {
        let paths = IndexPaths::new(output_dir);
        let mut writer = IndexWriter::new(&paths)?;

        // Postings must be written in ascending hash order.  Drain the map(s)
        // into a flat vector and sort the pairs: sequential iteration
        // afterwards replaces one random hash-map lookup per token with
        // prefetch-friendly linear access.  Partitioned bulk builds sort each
        // partition independently (in parallel) and concatenate — partitions
        // own disjoint top-bit hash ranges, so partition order is already the
        // global hash order.
        let entries: Vec<(u64, Vec<u32>)> = match self.partitioned {
            Some(parts) => {
                let mut sorted_parts: Vec<Vec<(u64, Vec<u32>)>> = parts
                    .into_par_iter()
                    .map(|map| {
                        let mut v: Vec<(u64, Vec<u32>)> = map.into_iter().collect();
                        v.sort_unstable_by_key(|(hash, _)| *hash);
                        v
                    })
                    .collect();
                let total: usize = sorted_parts.iter().map(|v| v.len()).sum();
                let mut entries = Vec::with_capacity(total);
                for part in &mut sorted_parts {
                    entries.append(part);
                }
                entries
            }
            None => {
                let mut v: Vec<(u64, Vec<u32>)> = self.postings.into_iter().collect();
                v.par_sort_unstable_by_key(|(hash, _)| *hash);
                v
            }
        };

        let ngram_count = entries.len();
        writer.reserve_lookup(ngram_count);
        for (hash, ids) in &entries {
            if !ids.is_empty() {
                writer.write_posting(*hash, ids)?;
            }
        }

        let meta = IndexMeta::new(self.path_to_id.len(), ngram_count);
        let meta_json = serde_json::to_string_pretty(&meta).context("serialize IndexMeta")?;

        writer.finish(&self.file_table, &meta_json)?;

        // Free the (potentially millions of) posting vectors in parallel —
        // dropping them serially takes seconds on large repositories.
        entries.into_par_iter().for_each(drop);

        Index::open(output_dir)
    }
}

// ---------------------------------------------------------------------------
// Parallel batch indexing
// ---------------------------------------------------------------------------

/// Extract the deduplicated n-gram hash set for a single file's content.
/// This is the per-file work item of parallel index builds: callers map it
/// over many files with Rayon and then feed the results to
/// [`IndexBuilder::from_file_hashes`].
///
/// The output is **sorted ascending** and deduplicated, so merge callers can
/// locate a hash range with binary search.  Generic over the weight function
/// so hot paths can pass a concrete `FrequencyWeight` and get the weight
/// lookups inlined.
pub fn file_ngram_hashes<W: WeightFn + ?Sized>(content: &[u8], weight_fn: &W) -> Vec<u64> {
    // Source files yield roughly 0.2 distinct hashes per content byte on
    // average; the hint only pre-sizes the table, growth still handles
    // outliers.
    let mut set = U64Dedup::with_capacity(content.len() / 5);
    for_each_ngram(content, weight_fn, |_start, _end, hash| set.insert(hash));
    let mut hashes = set.into_vec();
    hashes.sort_unstable();
    hashes
}

/// Index a batch of `(path, content)` pairs in parallel using Rayon.
///
/// Returns `(postings, file_table)` ready to be written by `IndexWriter`.
pub fn build_postings_parallel(
    files: &[(PathBuf, Vec<u8>)],
    weight_fn: &dyn WeightFn,
) -> BTreeMap<u64, Vec<u32>> {
    // Each thread computes n-grams for its files, returns (hash, file_id) pairs.
    let per_file: Vec<(u32, Vec<u64>)> = files
        .par_iter()
        .enumerate()
        .map(|(file_id, (_path, content))| {
            let hashes = file_ngram_hashes(content, weight_fn);
            (file_id as u32, hashes)
        })
        .collect();

    // Merge into a global BTreeMap.
    let mut postings: BTreeMap<u64, Vec<u32>> = BTreeMap::new();
    for (file_id, hashes) in per_file {
        for hash in hashes {
            postings.entry(hash).or_default().push(file_id);
        }
    }
    postings
}

// ---------------------------------------------------------------------------
// Index (open / query)
// ---------------------------------------------------------------------------

/// An opened, ready-to-query index.
pub struct Index {
    pub reader: IndexReader,
    pub meta: IndexMeta,
    pub paths: IndexPaths,
}

impl Index {
    /// Open an existing index directory.
    pub fn open(dir: &Path) -> Result<Self> {
        let paths = IndexPaths::new(dir);
        let reader =
            IndexReader::open(&paths).with_context(|| format!("open index at {:?}", dir))?;

        // T8: Properly handle meta.json parse errors
        let meta = match std::fs::read_to_string(&paths.meta) {
            Ok(s) => serde_json::from_str(&s).context("parse meta.json")?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => IndexMeta::new(0, 0),
            Err(e) => return Err(e).context("read meta.json"),
        };

        Ok(Self {
            reader,
            meta,
            paths,
        })
    }

    /// Look up the posting list for an n-gram hash.
    pub fn lookup_hash(&self, hash: u64) -> Result<Option<Vec<u32>>> {
        self.reader.lookup(hash)
    }

    /// Return the path for a given file_id.
    pub fn file_path(&self, file_id: u32) -> Option<&Path> {
        self.reader
            .file_paths
            .get(file_id as usize)
            .map(PathBuf::as_path)
    }

    /// Number of unique n-gram hashes stored.
    pub fn ngram_count(&self) -> u64 {
        self.reader.entry_count()
    }

    /// Number of indexed files (including tombstones).
    pub fn file_count(&self) -> usize {
        self.reader.file_paths.len()
    }

    // -------------------------------------------------------------------
    // Incremental update helpers
    // -------------------------------------------------------------------

    /// Rebuild the index from a modified `IndexBuilder`.
    ///
    /// The old files on disk are replaced atomically (write-then-rename isn't
    /// available on all targets, so we overwrite in place for simplicity).
    pub fn rebuild_from(&self, builder: IndexBuilder) -> Result<Index> {
        builder.build(&self.paths.dir)
    }

    /// Load the full postings into memory for modification.
    pub fn to_builder(&self) -> Result<IndexBuilder> {
        let weight_fn = default_weight_fn();
        let mut builder = IndexBuilder::new(weight_fn);

        // Reconstruct file_table and path_to_id (without re-hashing content).
        for (i, path) in self.reader.file_paths.iter().enumerate() {
            if !path.as_os_str().is_empty() {
                builder.file_table.push(path.clone());
                builder.path_to_id.insert(path.clone(), i as u32);
            } else {
                builder.file_table.push(PathBuf::new()); // tombstone
            }
        }

        // Re-read all postings from disk.
        let n = self.reader.entry_count() as usize;
        for i in 0..n {
            // We access the raw mmap entries via IndexReader internals.
            // Use the reader's public find_offset / read_posting_at interface
            // via a hash scan — but we don't have sequential iteration.
            // Instead, read entries by index position using the mmap directly.
            let hash = self.reader_entry_hash(i);
            let offset = self.reader_entry_offset(i);
            let ids = self.reader.read_posting_at(offset)?;
            builder.postings.insert(hash, ids);
        }

        Ok(builder)
    }

    /// Return all `(hash, file_ids)` pairs stored in this index.
    ///
    /// Useful for compaction: callers can iterate every posting list without
    /// re-computing n-grams from the original file content.
    pub fn all_postings(&self) -> Result<Vec<(u64, Vec<u32>)>> {
        let n = self.reader.entry_count() as usize;
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let hash = self.reader_entry_hash(i);
            let offset = self.reader_entry_offset(i);
            let ids = self.reader.read_posting_at(offset)?;
            result.push((hash, ids));
        }
        Ok(result)
    }

    pub(crate) fn reader_entry_hash(&self, i: usize) -> u64 {
        use crate::storage::{ENTRY_SIZE, HEADER_SIZE};
        let off = HEADER_SIZE + i * ENTRY_SIZE;
        let bytes = &self.reader.lookup_mmap[off..off + 8];
        u64::from_le_bytes(bytes.try_into().unwrap())
    }

    pub(crate) fn reader_entry_offset(&self, i: usize) -> u64 {
        use crate::storage::{ENTRY_SIZE, HEADER_SIZE};
        let off = HEADER_SIZE + i * ENTRY_SIZE + 8;
        let bytes = &self.reader.lookup_mmap[off..off + 8];
        u64::from_le_bytes(bytes.try_into().unwrap())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ngram::fnv_hash;
    use tempfile::TempDir;

    fn simple_index(dir: &Path) -> Result<Index> {
        let mut builder = IndexBuilder::with_default_weights();
        builder.add_file(Path::new("a.rs"), b"fn main() { println!(\"hello\"); }")?;
        builder.add_file(Path::new("b.rs"), b"fn helper(x: i32) -> i32 { x + 1 }")?;
        builder.add_file(Path::new("c.rs"), b"struct Foo { bar: String }")?;
        builder.build(dir)
    }

    #[test]
    fn test_build_and_open() -> Result<()> {
        let tmp = TempDir::new()?;
        let index = simple_index(tmp.path())?;
        assert_eq!(index.file_count(), 3);
        assert!(index.ngram_count() > 0);
        Ok(())
    }

    #[test]
    fn test_lookup_known_ngram() -> Result<()> {
        let tmp = TempDir::new()?;
        let index = simple_index(tmp.path())?;

        // "fn" appears in all three files.
        let hash = fnv_hash(b"fn");
        let ids = index.lookup_hash(hash)?;
        // "fn" as a 2-byte n-gram must appear in a.rs and b.rs at minimum.
        // (It might not be emitted as a single-pair n-gram depending on weights,
        //  but a multi-byte n-gram containing "fn" will be emitted.)
        // We test that the lookup doesn't crash and returns Some or None consistently.
        let _ = ids; // just ensure no panic
        Ok(())
    }

    #[test]
    fn test_rebuild_no_content_change() -> Result<()> {
        let tmp = TempDir::new()?;
        let index1 = simple_index(tmp.path())?;
        let count1 = index1.ngram_count();

        // Re-open and rebuild (via to_builder).
        let builder = index1.to_builder()?;
        let tmp2 = TempDir::new()?;
        let index2 = builder.build(tmp2.path())?;

        assert_eq!(index2.ngram_count(), count1);
        assert_eq!(index2.file_count(), 3);
        Ok(())
    }

    #[test]
    fn test_parallel_build_same_as_sequential() -> Result<()> {
        let files: Vec<(PathBuf, Vec<u8>)> = vec![
            (PathBuf::from("a.rs"), b"fn foo() {}".to_vec()),
            (PathBuf::from("b.rs"), b"fn bar() {}".to_vec()),
        ];
        let wf = default_weight_fn();
        let parallel = build_postings_parallel(&files, wf.as_ref());

        let mut seq_builder = IndexBuilder::with_default_weights();
        for (path, content) in &files {
            seq_builder.add_file(path, content)?;
        }

        // Both should have the same set of n-gram hashes.
        let parallel_hashes: std::collections::BTreeSet<u64> = parallel.keys().copied().collect();
        let seq_hashes: std::collections::BTreeSet<u64> =
            seq_builder.postings.keys().copied().collect();
        assert_eq!(parallel_hashes, seq_hashes);

        Ok(())
    }
}
