//! Index file format and I/O.
//!
//! # Directory layout
//!
//! ```text
//! .sngi-store/                    ← INDEX_STORE_DIR (IndexStorePaths root)
//!   state.json                    ← SegmentManager state
//!   config.json                   ← store-level configuration
//!   segments/
//!     segment_0/                  ← one segment per IndexPaths
//!       lookup.bin      – sorted (hash: u64, offset: u64) pairs  → mmap'd
//!       postings.bin    – varint-encoded posting lists on disk
//!       files.txt       – newline-separated file paths (file_id = line number)
//!       meta.json       – index metadata
//!       deletion.bin    – per-segment DeletionBitmap (optional)
//!     segment_1/
//!       …
//!   branches/                     ← per-branch manifests (future)
//! ```
//!
//! # lookup.bin format
//!
//! ```text
//! Header (24 bytes):
//!   [0..8]   magic   = 0x534E47_494C4B5550  ("SNGILKUP" as u64 LE)
//!   [8..12]  version = 1u32 LE
//!   [12..16] flags   = 0u32 (reserved)
//!   [16..24] count   = number of entries (u64 LE)
//!
//! Entries (count × 16 bytes, sorted by hash):
//!   [0..8]   hash    u64 LE
//!   [8..16]  offset  u64 LE  → byte offset into postings.bin
//! ```
//!
//! # postings.bin format
//!
//! Each posting list block starts at the offset stored in lookup.bin:
//! ```text
//!   length: u32 LE   – number of file_id entries
//!   file_ids: [u32 LE; length]
//! ```

use std::fs::{self, File, OpenOptions};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{bail, Context, Result};
use memmap2::Mmap;

// ---------------------------------------------------------------------------
// T9: Atomic write utility
// ---------------------------------------------------------------------------

/// Write bytes to a file atomically using write-tmp-then-rename.
/// This ensures readers never see partial writes.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp_path = path.with_extension("tmp");
    let mut f =
        BufWriter::new(File::create(&tmp_path).with_context(|| format!("create {:?}", tmp_path))?);
    f.write_all(bytes)?;
    f.flush()?;
    // Ensure data is synced to disk before rename
    f.into_inner()?.sync_all()?;
    std::fs::rename(&tmp_path, path)
        .with_context(|| format!("rename {:?} to {:?}", tmp_path, path))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Magic number stored at the top of `lookup.bin`.
/// ASCII bytes "SNGILKUP" in little-endian u64.
pub const MAGIC: u64 = 0x5055_4b4c_4947_4e53;
pub const VERSION: u32 = 1;

/// Size of the lookup.bin header in bytes.
pub const HEADER_SIZE: usize = 24;
/// Size of each lookup entry in bytes.
pub const ENTRY_SIZE: usize = 16;

/// Name of the index store sub-directory inside a repo.
pub const INDEX_STORE_DIR: &str = ".sngi-store";

/// Directories that should always be skipped during indexing, regardless of
/// .gitignore configuration. These are either VCS internals, dependency
/// directories, build outputs, or IDE metadata that provide no search value.
pub const ALWAYS_SKIP_DIRS: &[&str] = &[
    // VCS internals
    ".git",
    ".hg",
    ".svn",
    // Index store directories (current and legacy)
    ".sngi-store",
    ".sngi",
    // Dependencies
    "node_modules",
    "vendor",
    "bower_components",
    // Build outputs
    "target",
    "dist",
    "build",
    "out",
    ".next",
    ".nuxt",
    // IDE / editor
    ".idea",
    ".vscode",
    ".eclipse",
    ".settings",
    // Caches
    ".cache",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    // Python environments
    ".venv",
    "venv",
    ".tox",
    // Package manager caches
    ".npm",
    ".yarn",
    ".pnpm-store",
    // Coverage / test output
    "coverage",
    ".nyc_output",
    "htmlcov",
];

/// Check if a directory name matches any entry in [`ALWAYS_SKIP_DIRS`].
pub fn should_skip_dir(dir_name: &str) -> bool {
    ALWAYS_SKIP_DIRS.contains(&dir_name)
}

// ---------------------------------------------------------------------------
// External index directory support
// ---------------------------------------------------------------------------

static EXTERNAL_INDEX_BASE: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Set the external index base directory programmatically.
/// This overrides the `SNGS_INDEX_DIR` environment variable.
/// Call once at CLI startup before any index operations.
pub fn set_external_index_base(path: Option<PathBuf>) {
    let _ = EXTERNAL_INDEX_BASE.set(path);
}

/// Build a stable subdirectory name under `base` for the given repository root.
/// Uses the canonical repository path hashed to a 16-character hex string.
fn external_index_subdir(repo_root: &Path, base: &Path) -> PathBuf {
    let canonical = repo_root
        .canonicalize()
        .unwrap_or_else(|_| repo_root.to_path_buf());
    let mut hasher = DefaultHasher::new();
    canonical.as_os_str().hash(&mut hasher);
    let hash = hasher.finish();
    base.join(format!("{:016x}", hash))
}

/// Resolve the index store directory for a given repository root.
///
/// Priority:
/// 1. Explicitly set base via [`set_external_index_base()`]
/// 2. `SNGS_INDEX_DIR` environment variable
/// 3. Default: `{repo_root}/.sngi-store`
pub fn resolve_index_dir(repo_root: &Path) -> PathBuf {
    // 1. Explicitly set base (highest priority)
    if let Some(Some(base)) = EXTERNAL_INDEX_BASE.get() {
        return external_index_subdir(repo_root, base).join(INDEX_STORE_DIR);
    }

    // 2. Environment variable
    if let Ok(base_str) = std::env::var("SNGS_INDEX_DIR") {
        let base = PathBuf::from(base_str);
        return external_index_subdir(repo_root, &base).join(INDEX_STORE_DIR);
    }

    // 3. Default: inside repo
    repo_root.join(INDEX_STORE_DIR)
}

// ---------------------------------------------------------------------------
// Index paths
// ---------------------------------------------------------------------------

/// Top-level path layout for the `.sngi-store/` directory tree.
///
/// Use this to locate the store root, segment subdirectory, branch manifests,
/// and the top-level state/config files.
#[derive(Debug, Clone)]
pub struct IndexStorePaths {
    /// `.sngi-store/`
    pub root: PathBuf,
    /// `.sngi-store/segments/`
    pub segments_dir: PathBuf,
    /// `.sngi-store/branches/`
    pub branches_dir: PathBuf,
    /// `.sngi-store/state.json`
    pub state_json: PathBuf,
    /// `.sngi-store/config.json`
    pub config_json: PathBuf,
    /// `.sngi-store/.lock` - F6: file lock
    pub lock_file: PathBuf,
}

impl IndexStorePaths {
    /// Build the canonical path set for the given `repo_root`.
    pub fn new(repo_root: &Path) -> Self {
        let root = resolve_index_dir(repo_root);
        Self {
            segments_dir: root.join("segments"),
            branches_dir: root.join("branches"),
            state_json: root.join("state.json"),
            config_json: root.join("config.json"),
            lock_file: root.join(".lock"),
            root,
        }
    }

    /// `true` when `state.json` exists (i.e., a store has been initialised).
    pub fn exists(&self) -> bool {
        self.state_json.exists()
    }
}

/// Canonical paths for a single segment directory
/// (`lookup.bin`, `postings.bin`, `files.txt`, `meta.json`).
#[derive(Debug, Clone)]
pub struct IndexPaths {
    pub dir: PathBuf,
    pub lookup: PathBuf,
    pub postings: PathBuf,
    pub files: PathBuf,
    pub meta: PathBuf,
}

impl IndexPaths {
    pub fn new(dir: &Path) -> Self {
        Self {
            lookup: dir.join("lookup.bin"),
            postings: dir.join("postings.bin"),
            files: dir.join("files.txt"),
            meta: dir.join("meta.json"),
            dir: dir.to_path_buf(),
        }
    }

    pub fn exists(&self) -> bool {
        self.lookup.exists() && self.postings.exists()
    }
}

// ---------------------------------------------------------------------------
// IndexWriter
// ---------------------------------------------------------------------------

/// Capacity of the `BufWriter`s used for `lookup.bin` / `postings.bin`.
const WRITE_BUF_CAPACITY: usize = 8 << 20;

/// Writes the three index files atomically.
///
/// Usage:
/// ```ignore
/// let mut writer = IndexWriter::new(&paths)?;
/// writer.set_files(file_paths);
/// for (hash, file_ids) in postings { writer.write_posting(hash, file_ids)?; }
/// writer.finish()?;
/// ```
///
/// Entries must be written in **ascending hash order** (the caller sorts them).
pub struct IndexWriter {
    lookup_buf: BufWriter<File>,
    postings_buf: BufWriter<File>,
    /// Running byte offset into postings.bin.
    postings_offset: u64,
    /// (hash, postings_offset) pairs accumulated for the lookup table.
    lookup_entries: Vec<(u64, u64)>,
    paths: IndexPaths,
}

impl IndexWriter {
    /// Create (or truncate) the index files and write the lookup header.
    pub fn new(paths: &IndexPaths) -> Result<Self> {
        fs::create_dir_all(&paths.dir)
            .with_context(|| format!("create index dir {:?}", paths.dir))?;

        let lookup_file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&paths.lookup)
            .with_context(|| format!("open {:?}", paths.lookup))?;

        let postings_file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&paths.postings)
            .with_context(|| format!("open {:?}", paths.postings))?;

        Ok(Self {
            // Large write buffers: postings.bin reaches gigabytes on big
            // repositories, and the default 8 KiB buffer turns that into
            // hundreds of thousands of write(2) calls.
            lookup_buf: BufWriter::with_capacity(WRITE_BUF_CAPACITY, lookup_file),
            postings_buf: BufWriter::with_capacity(WRITE_BUF_CAPACITY, postings_file),
            postings_offset: 0,
            lookup_entries: Vec::new(),
            paths: paths.clone(),
        })
    }

    /// Pre-allocate the lookup table for `tokens` posting lists, avoiding
    /// repeated realloc-and-copy while they are appended in hash order.
    pub fn reserve_lookup(&mut self, tokens: usize) {
        self.lookup_entries.reserve(tokens);
    }

    /// Write one posting list.  Must be called in ascending `hash` order.
    pub fn write_posting(&mut self, hash: u64, file_ids: &[u32]) -> Result<()> {
        let offset = self.postings_offset;
        self.lookup_entries.push((hash, offset));

        // Write: length (u32 LE) + file_ids (u32 LE each).
        let len = file_ids.len() as u32;
        self.postings_buf.write_all(&len.to_le_bytes())?;
        self.write_u32_slice(file_ids)?;

        self.postings_offset += 4 + file_ids.len() as u64 * 4;
        Ok(())
    }

    /// Write a slice of u32s as little-endian bytes in one call.
    fn write_u32_slice(&mut self, ids: &[u32]) -> Result<()> {
        #[cfg(target_endian = "little")]
        {
            // SAFETY: u8 may view any object's bytes; u32 has no padding, and
            // on little-endian targets the in-memory layout already matches
            // the on-disk LE format.  The slice does not outlive this call.
            let bytes =
                unsafe { std::slice::from_raw_parts(ids.as_ptr() as *const u8, ids.len() * 4) };
            self.postings_buf.write_all(bytes)?;
            Ok(())
        }
        #[cfg(not(target_endian = "little"))]
        {
            for &id in ids {
                self.postings_buf.write_all(&id.to_le_bytes())?;
            }
            Ok(())
        }
    }

    /// Write all lookup entries `(hash: u64 LE, offset: u64 LE)` in one call.
    fn write_lookup_entries(&mut self) -> Result<()> {
        // (u64, u64) has no padding; layout is two consecutive u64s.
        const _: () = assert!(std::mem::size_of::<(u64, u64)>() == ENTRY_SIZE);
        #[cfg(target_endian = "little")]
        {
            // SAFETY: u8 may view any object's bytes; (u64, u64) has no
            // padding (asserted above), and on little-endian targets the
            // in-memory layout matches the on-disk LE format.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    self.lookup_entries.as_ptr() as *const u8,
                    self.lookup_entries.len() * ENTRY_SIZE,
                )
            };
            self.lookup_buf.write_all(bytes)?;
            Ok(())
        }
        #[cfg(not(target_endian = "little"))]
        {
            for (hash, offset) in &self.lookup_entries {
                self.lookup_buf.write_all(&hash.to_le_bytes())?;
                self.lookup_buf.write_all(&offset.to_le_bytes())?;
            }
            Ok(())
        }
    }

    /// Write the file path table and finalise the lookup header.
    pub fn finish(mut self, file_paths: &[PathBuf], meta_json: &str) -> Result<()> {
        // Flush postings.
        self.postings_buf.flush()?;

        // Write lookup header + entries.
        let count = self.lookup_entries.len() as u64;
        self.lookup_buf.write_all(&MAGIC.to_le_bytes())?;
        self.lookup_buf.write_all(&VERSION.to_le_bytes())?;
        self.lookup_buf.write_all(&0u32.to_le_bytes())?; // flags
        self.lookup_buf.write_all(&count.to_le_bytes())?;
        self.write_lookup_entries()?;
        self.lookup_buf.flush()?;

        // T2: Write files.bin in binary format to support tombstones.
        // Format: for each file, write length as u32 LE, then bytes.
        // Length of 0 means tombstone (empty slot).
        let mut f = BufWriter::new(
            File::create(&self.paths.files)
                .with_context(|| format!("create {:?}", self.paths.files))?,
        );
        for path in file_paths {
            let bytes = path.as_os_str().as_encoded_bytes();
            let len = bytes.len() as u32;
            f.write_all(&len.to_le_bytes())?;
            if len > 0 {
                f.write_all(bytes)?;
            }
        }
        f.flush()?;

        // T9: Write meta.json atomically.
        write_atomic(&self.paths.meta, meta_json.as_bytes())?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// IndexReader
// ---------------------------------------------------------------------------

/// Read file paths from files.txt, supporting both binary and legacy newline formats.
///
/// Binary format (current): for each file, u32 LE length + bytes. Length 0 = tombstone.
/// Legacy format: newline-separated UTF-8 text (one path per line).
fn read_file_paths_with_compat(files_bytes: &[u8], files_path: &Path) -> Result<Vec<PathBuf>> {
    // Heuristic: if the file looks like valid UTF-8 and doesn't start with a plausible
    // length prefix (0..2^20), treat as legacy newline format.
    if files_bytes.len() >= 4 {
        let len = u32::from_le_bytes([
            files_bytes[0],
            files_bytes[1],
            files_bytes[2],
            files_bytes[3],
        ]) as usize;
        // If len is reasonable (0 for tombstone or < 1MB) and the binary parse succeeds, use binary format
        if len < (1 << 20) && try_read_binary_format(files_bytes).is_ok() {
            return try_read_binary_format(files_bytes);
        }
    }

    // Try legacy newline format
    if let Ok(text) = std::str::from_utf8(files_bytes) {
        let paths: Vec<PathBuf> = text.lines().map(PathBuf::from).collect();
        if !paths.is_empty() {
            eprintln!(
                "warning: {:?} is in legacy newline format. It will be migrated on next write.",
                files_path
            );
            return Ok(paths);
        }
    }

    // Fall back to binary format (will error if invalid)
    try_read_binary_format(files_bytes)
}

/// Try to parse files.txt as binary format.
fn try_read_binary_format(files_bytes: &[u8]) -> Result<Vec<PathBuf>> {
    let mut file_paths = Vec::new();
    let mut offset = 0;

    while offset + 4 <= files_bytes.len() {
        let len = u32::from_le_bytes([
            files_bytes[offset],
            files_bytes[offset + 1],
            files_bytes[offset + 2],
            files_bytes[offset + 3],
        ]) as usize;
        offset += 4;

        if len == 0 {
            // Tombstone: empty slot
            file_paths.push(PathBuf::new());
        } else {
            if offset + len > files_bytes.len() {
                bail!("files.txt truncated: expected {} more bytes", len);
            }
            let path_bytes = &files_bytes[offset..offset + len];
            #[cfg(unix)]
            let path = {
                use std::os::unix::ffi::OsStrExt;
                PathBuf::from(std::ffi::OsStr::from_bytes(path_bytes))
            };
            #[cfg(not(unix))]
            let path = {
                let path_str = String::from_utf8_lossy(path_bytes);
                PathBuf::from(path_str.as_ref())
            };
            file_paths.push(path);
            offset += len;
        }
    }

    Ok(file_paths)
}

/// Memory-maps `lookup.bin` for fast binary search; reads `postings.bin` on
/// demand via memory-mapped access.
pub struct IndexReader {
    /// Memory-mapped lookup.bin.
    pub(crate) lookup_mmap: Mmap,
    /// Number of entries in the lookup table.
    entry_count: u64,
    /// T22: Memory-mapped postings.bin for zero-copy reads.
    postings_mmap: Mmap,
    /// File path table (file_id → PathBuf).
    pub file_paths: Vec<PathBuf>,
}

impl IndexReader {
    /// Open an existing index.
    pub fn open(paths: &IndexPaths) -> Result<Self> {
        // Open and mmap lookup.bin.
        let lookup_file =
            File::open(&paths.lookup).with_context(|| format!("open {:?}", paths.lookup))?;
        // SAFETY: We treat the mmap as read-only and never write to it.
        let lookup_mmap = unsafe { Mmap::map(&lookup_file) }
            .with_context(|| format!("mmap {:?}", paths.lookup))?;

        // Validate header.
        if lookup_mmap.len() < HEADER_SIZE {
            bail!("lookup.bin too small: {} bytes", lookup_mmap.len());
        }
        let magic = u64::from_le_bytes(lookup_mmap[0..8].try_into().unwrap());
        if magic != MAGIC {
            bail!("lookup.bin has wrong magic: {:#x}", magic);
        }
        let version = u32::from_le_bytes(lookup_mmap[8..12].try_into().unwrap());
        if version != VERSION {
            bail!("lookup.bin version mismatch: expected {VERSION}, got {version}");
        }
        let entry_count = u64::from_le_bytes(lookup_mmap[16..24].try_into().unwrap());

        // Sanity check size.
        let expected_size = HEADER_SIZE + entry_count as usize * ENTRY_SIZE;
        if lookup_mmap.len() < expected_size {
            bail!(
                "lookup.bin too small for {entry_count} entries: have {} bytes, need {}",
                lookup_mmap.len(),
                expected_size
            );
        }

        // T22: Memory-map postings.bin for zero-copy reads
        let postings_file =
            File::open(&paths.postings).with_context(|| format!("open {:?}", paths.postings))?;
        let postings_mmap = unsafe { Mmap::map(&postings_file) }
            .with_context(|| format!("mmap {:?}", paths.postings))?;

        // T2 + F4: Read file paths from binary or legacy newline format.
        // Try binary format first (len:u32 LE + bytes), fall back to newline-separated text.
        let files_bytes =
            fs::read(&paths.files).with_context(|| format!("read {:?}", paths.files))?;
        let file_paths = read_file_paths_with_compat(&files_bytes, &paths.files)?;

        Ok(Self {
            lookup_mmap,
            entry_count,
            postings_mmap,
            file_paths,
        })
    }

    /// Binary-search the lookup table for `hash`.
    /// Returns the posting list offset if found, else `None`.
    pub fn find_offset(&self, hash: u64) -> Option<u64> {
        let n = self.entry_count as usize;
        if n == 0 {
            return None;
        }
        // Binary search over entries in the mmap.
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let entry_hash = self.entry_hash(mid);
            match entry_hash.cmp(&hash) {
                std::cmp::Ordering::Equal => return Some(self.entry_offset(mid)),
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
            }
        }
        None
    }

    /// Read the posting list at the given byte offset in postings.bin.
    /// T22: Use mmap for zero-copy reads instead of pread syscalls.
    pub fn read_posting_at(&self, offset: u64) -> Result<Vec<u32>> {
        let offset = offset as usize;
        if offset + 4 > self.postings_mmap.len() {
            bail!("posting offset {} exceeds postings size", offset);
        }

        // Read length from mmap
        let len_bytes: [u8; 4] = self.postings_mmap[offset..offset + 4].try_into().unwrap();
        let len = u32::from_le_bytes(len_bytes) as usize;

        // Read file IDs from mmap
        let data_offset = offset + 4;
        let data_end = data_offset + len * 4;
        if data_end > self.postings_mmap.len() {
            bail!("posting list at offset {} exceeds postings size", offset);
        }

        let mut ids = Vec::with_capacity(len);
        for chunk in self.postings_mmap[data_offset..data_end].as_chunks::<4>().0 {
            ids.push(u32::from_le_bytes(*chunk));
        }
        Ok(ids)
    }

    /// Convenience: look up `hash` and return its posting list.
    pub fn lookup(&self, hash: u64) -> Result<Option<Vec<u32>>> {
        match self.find_offset(hash) {
            None => Ok(None),
            Some(off) => Ok(Some(self.read_posting_at(off)?)),
        }
    }

    #[inline]
    fn entry_hash(&self, i: usize) -> u64 {
        let off = HEADER_SIZE + i * ENTRY_SIZE;
        u64::from_le_bytes(self.lookup_mmap[off..off + 8].try_into().unwrap())
    }

    #[inline]
    fn entry_offset(&self, i: usize) -> u64 {
        let off = HEADER_SIZE + i * ENTRY_SIZE + 8;
        u64::from_le_bytes(self.lookup_mmap[off..off + 8].try_into().unwrap())
    }

    pub fn entry_count(&self) -> u64 {
        self.entry_count
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_test_index(dir: &Path) -> Result<IndexPaths> {
        let paths = IndexPaths::new(dir);
        let mut writer = IndexWriter::new(&paths)?;

        // Three n-gram hashes with simple posting lists.
        let entries: &[(u64, &[u32])] = &[(10, &[0, 1, 2]), (20, &[1, 3]), (30, &[0, 2, 4])];
        for (hash, ids) in entries {
            writer.write_posting(*hash, ids)?;
        }
        let paths_vec: Vec<PathBuf> = (0..5)
            .map(|i| PathBuf::from(format!("file{i}.txt")))
            .collect();
        writer.finish(&paths_vec, r#"{"version":1}"#)?;
        Ok(paths)
    }

    #[test]
    fn test_write_read_roundtrip() -> Result<()> {
        let tmp = TempDir::new()?;
        let paths = make_test_index(tmp.path())?;

        let reader = IndexReader::open(&paths)?;
        assert_eq!(reader.entry_count(), 3);

        // Lookup existing hashes.
        let ids = reader.lookup(10)?.unwrap();
        assert_eq!(ids, vec![0, 1, 2]);

        let ids = reader.lookup(20)?.unwrap();
        assert_eq!(ids, vec![1, 3]);

        let ids = reader.lookup(30)?.unwrap();
        assert_eq!(ids, vec![0, 2, 4]);

        // Miss.
        assert!(reader.lookup(99)?.is_none());

        // File paths.
        assert_eq!(reader.file_paths.len(), 5);
        assert_eq!(reader.file_paths[0], PathBuf::from("file0.txt"));

        Ok(())
    }

    #[test]
    fn test_binary_search_first_last() -> Result<()> {
        let tmp = TempDir::new()?;
        let paths = IndexPaths::new(tmp.path());
        let mut writer = IndexWriter::new(&paths)?;

        // 100 entries with ascending hashes.
        for i in 0u64..100 {
            writer.write_posting(i * 7, &[i as u32])?;
        }
        writer.finish(&[], r#"{}"#)?;

        let reader = IndexReader::open(&paths)?;
        // First entry.
        let ids = reader.lookup(0)?.unwrap();
        assert_eq!(ids, vec![0]);
        // Last entry.
        let ids = reader.lookup(99 * 7)?.unwrap();
        assert_eq!(ids, vec![99]);
        // Between entries.
        assert!(reader.lookup(1)?.is_none());

        Ok(())
    }

    #[test]
    fn test_should_skip_dir() {
        // VCS internals
        assert!(should_skip_dir(".git"));
        assert!(should_skip_dir(".hg"));
        assert!(should_skip_dir(".svn"));
        // Index store dir
        assert!(should_skip_dir(".sngi-store"));
        // Dependencies
        assert!(should_skip_dir("node_modules"));
        assert!(should_skip_dir("vendor"));
        // Build outputs
        assert!(should_skip_dir("target"));
        assert!(should_skip_dir("dist"));
        // Not in the list
        assert!(!should_skip_dir("src"));
        assert!(!should_skip_dir("lib"));
        assert!(!should_skip_dir(".github"));
        assert!(!should_skip_dir(""));
    }

    #[test]
    fn test_empty_index() -> Result<()> {
        let tmp = TempDir::new()?;
        let paths = IndexPaths::new(tmp.path());
        let writer = IndexWriter::new(&paths)?;
        writer.finish(&[], r#"{}"#)?;

        let reader = IndexReader::open(&paths)?;
        assert_eq!(reader.entry_count(), 0);
        assert!(reader.lookup(42)?.is_none());

        Ok(())
    }
}
