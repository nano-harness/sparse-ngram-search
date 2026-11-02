//! Query engine: regex → n-gram lookup → candidate files → exact match.
//!
//! # Pipeline
//!
//! 1. **Decompose** the pattern into a [`QueryAtom`] tree of n-gram hashes
//!    via [`regex_decompose::decompose_regex`].
//! 2. **Resolve** each leaf hash against the inverted index to get a set of
//!    candidate file_ids.
//! 3. **Combine** sets: `And` → intersect, `Or` → union.
//! 4. **Verify** each candidate file with the full regex engine to eliminate
//!    false positives.  (The caller must supply file contents, or provide a
//!    content loader callback.)
//!
//! The result is a [`Vec<SearchResult>`].  Candidates whose content cannot be
//! loaded are excluded rather than reported unverified, so results contain no
//! false positives from unreadable or deleted files.

use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::{Context, Result};
use rayon::prelude::*;
use regex::bytes::Regex;

use crate::frequency_table::default_weight_fn;
use crate::index::Index;
use crate::ngram::{build_covering, WeightFn};
use crate::regex_decompose::{decompose_regex, QueryAtom};

/// Candidate count above which exact-match verification runs in parallel.
/// Each verification reads a file from disk, so large candidate sets are
/// dominated by per-file syscalls and benefit from all cores.
const PAR_VERIFY_MIN: usize = 64;

/// Run `verify` over every candidate, collecting the matches.  Large sets
/// are processed with Rayon (order is preserved); small sets stay sequential
/// to avoid task overhead.
fn verify_candidates<V>(candidates: Vec<u32>, verify: V) -> Vec<SearchResult>
where
    V: Fn(u32) -> Option<SearchResult> + Sync + Send,
{
    if candidates.len() < PAR_VERIFY_MIN {
        candidates.into_iter().filter_map(verify).collect()
    } else {
        candidates.into_par_iter().filter_map(verify).collect()
    }
}

// ---------------------------------------------------------------------------
// SearchResult
// ---------------------------------------------------------------------------

/// One match returned by the query engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchResult {
    /// Numeric file identifier in the index.
    pub file_id: u32,
    /// Path as recorded in the index.
    pub path: PathBuf,
    /// Byte offsets of all regex matches within the file (if verification was
    /// performed and match positions were requested).
    pub match_offsets: Vec<usize>,
}

// ---------------------------------------------------------------------------
// MatchDetail
// ---------------------------------------------------------------------------

/// Line-level detail for a single match within a file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchDetail {
    /// 1-based line number of the match.
    pub line_number: usize,
    /// 1-based column (byte offset within the line).
    pub column: usize,
    /// Full content of the line containing the match (without the trailing newline).
    pub line_content: String,
}

/// Convert a byte `offset` within `content` into a [`MatchDetail`].
///
/// If `offset` exceeds `content.len()` it is clamped to `content.len()`.
/// The result is always valid; if `offset` points past the last newline the
/// last (partial) line is returned.
pub fn offset_to_match_detail(content: &[u8], offset: usize) -> MatchDetail {
    let offset = offset.min(content.len());

    // Count newlines before `offset` to get the 1-based line number.
    let line_number = content[..offset].iter().filter(|&&b| b == b'\n').count() + 1;

    // Find the start of the current line.
    let line_start = content[..offset]
        .iter()
        .rposition(|&b| b == b'\n')
        .map(|p| p + 1)
        .unwrap_or(0);

    // Find the end of the current line (exclusive, not including the '\n').
    let line_end = content[offset..]
        .iter()
        .position(|&b| b == b'\n')
        .map(|p| offset + p)
        .unwrap_or(content.len());

    let line_bytes = &content[line_start..line_end];
    // F11: Strip trailing CRLF (\r)
    let line_bytes = if line_bytes.ends_with(b"\r") {
        &line_bytes[..line_bytes.len() - 1]
    } else {
        line_bytes
    };
    let line_content = String::from_utf8_lossy(line_bytes).into_owned();
    let column = offset - line_start + 1;

    MatchDetail {
        line_number,
        column,
        line_content,
    }
}

/// Extract context lines around `line_number` (1-based) from `content`.
///
/// Returns a list of `(line_number, line_content)` pairs covering
/// `[line_number - context_before, line_number + context_after]` (clamped to
/// the file boundaries).  Lines are returned in ascending order.
pub fn extract_context_lines(
    content: &[u8],
    line_number: usize,
    context_before: usize,
    context_after: usize,
) -> Vec<(usize, String)> {
    // Split content into lines (without trailing '\n').
    let lines: Vec<&[u8]> = content.split(|&b| b == b'\n').collect();
    let total = lines.len();

    if total == 0 || line_number == 0 {
        return vec![];
    }

    // Convert to 0-based index.
    let idx = line_number.saturating_sub(1).min(total - 1);
    let first = idx.saturating_sub(context_before);
    let last = (idx + context_after).min(total - 1);

    (first..=last)
        .map(|i| {
            // F11: Strip trailing CRLF (\r)
            let line = lines[i];
            let line = if line.ends_with(b"\r") {
                &line[..line.len() - 1]
            } else {
                line
            };
            let content = String::from_utf8_lossy(line).into_owned();
            (i + 1, content)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Galloping intersection
// ---------------------------------------------------------------------------

/// Intersect two sorted posting lists using galloping search (exponential search).
///
/// This is efficient when one list is much smaller than the other.
/// We iterate through the smaller list and search for each element in the larger list
/// using exponential search followed by binary search.
fn galloping_intersect(a: &[u32], b: &[u32]) -> Vec<u32> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::new();
    let mut b_pos = 0;

    for &elem in a {
        // Galloping search for elem in b[b_pos..]
        if b_pos >= b.len() {
            break;
        }

        // Skip elements in b that are less than elem
        while b_pos < b.len() && b[b_pos] < elem {
            // Exponential search to find a position >= elem
            let mut step = 1;
            while b_pos + step < b.len() && b[b_pos + step] < elem {
                b_pos += step;
                step *= 2;
            }

            // Binary search in the remaining range
            let search_start = b_pos;
            let search_end = (b_pos + step + 1).min(b.len());
            match b[search_start..search_end].binary_search(&elem) {
                Ok(idx) => {
                    b_pos = search_start + idx;
                    break;
                }
                Err(idx) => {
                    b_pos = search_start + idx;
                }
            }
        }

        // Check if we found elem
        if b_pos < b.len() && b[b_pos] == elem {
            result.push(elem);
            b_pos += 1;
        }
    }

    result
}

// ---------------------------------------------------------------------------
// QueryEngine
// ---------------------------------------------------------------------------

/// An immutable, borrowing view over an [`Index`] that executes queries.
pub struct QueryEngine<'a> {
    index: &'a Index,
    weight_fn: Box<dyn WeightFn>,
}

impl<'a> QueryEngine<'a> {
    pub fn new(index: &'a Index) -> Self {
        Self {
            index,
            weight_fn: default_weight_fn(),
        }
    }

    pub fn with_weight_fn(index: &'a Index, weight_fn: Box<dyn WeightFn>) -> Self {
        Self { index, weight_fn }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Search for files matching the given regex pattern.
    ///
    /// `load_content` is called for each candidate file to retrieve its bytes
    /// for exact-match verification.  If `None` is returned for a file (the
    /// content could not be read), the candidate is *excluded* — including an
    /// unverifiable candidate would turn deleted files and transient I/O
    /// errors into false positives.  Large candidate sets are verified in
    /// parallel, so the loader must be `Sync`.
    pub fn search<F>(&self, pattern: &str, load_content: F) -> Result<Vec<SearchResult>>
    where
        F: Fn(u32, &std::path::Path) -> Option<Vec<u8>> + Sync,
    {
        self.search_impl(pattern, load_content, true)
    }

    /// Like [`Self::search`], but stops verification at the first match per
    /// file and returns empty `match_offsets`.  Use for files-only listing
    /// where match positions are never displayed.
    pub fn search_files_only<F>(&self, pattern: &str, load_content: F) -> Result<Vec<SearchResult>>
    where
        F: Fn(u32, &std::path::Path) -> Option<Vec<u8>> + Sync,
    {
        self.search_impl(pattern, load_content, false)
    }

    fn search_impl<F>(
        &self,
        pattern: &str,
        load_content: F,
        collect_offsets: bool,
    ) -> Result<Vec<SearchResult>>
    where
        F: Fn(u32, &std::path::Path) -> Option<Vec<u8>> + Sync,
    {
        let atom = decompose_regex(pattern, self.weight_fn.as_ref())
            .with_context(|| format!("decompose regex: {pattern}"))?;

        let candidates = self.resolve_atom(&atom)?;

        if candidates.is_empty() {
            return Ok(vec![]);
        }

        // Compile the regex for exact verification.
        let re = Regex::new(pattern).with_context(|| format!("compile regex: {pattern}"))?;

        let verify = |file_id: u32| {
            let path = self.index.file_path(file_id)?.to_path_buf();
            let content = load_content(file_id, &path)?;
            if !collect_offsets {
                // Files-only mode: first match is enough.
                return if re.is_match(&content) {
                    Some(SearchResult {
                        file_id,
                        path,
                        match_offsets: vec![],
                    })
                } else {
                    None
                };
            }
            let offsets: Vec<usize> = re.find_iter(&content).map(|m| m.start()).collect();
            if offsets.is_empty() {
                None
            } else {
                Some(SearchResult {
                    file_id,
                    path,
                    match_offsets: offsets,
                })
            }
        };
        Ok(verify_candidates(candidates.into_iter().collect(), verify))
    }

    /// Search for an exact literal string (faster path: no regex parsing).
    pub fn search_literal<F>(&self, literal: &str, load_content: F) -> Result<Vec<SearchResult>>
    where
        F: Fn(u32, &std::path::Path) -> Option<Vec<u8>> + Sync,
    {
        let bytes = literal.as_bytes();
        let ngrams = build_covering(bytes, self.weight_fn.as_ref());

        if ngrams.is_empty() {
            // Literal too short to index — scan all files.
            return self.scan_all(literal, &load_content);
        }

        // T23: Use galloping intersect with size-based sorting
        let hashes: Vec<u64> = ngrams.iter().map(|ng| ng.hash()).collect();
        let candidates = self.intersect_hashes(&hashes)?;
        self.verify_literal(literal, candidates, &load_content)
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Intersect posting lists for a set of hashes using galloping search.
    /// Sorts posting lists by size (smallest first) and short-circuits on empty.
    fn intersect_hashes(&self, hashes: &[u64]) -> Result<BTreeSet<u32>> {
        if hashes.is_empty() {
            let n = self.index.file_count();
            return Ok((0..n as u32).collect());
        }

        // Resolve every hash once, up front (previously each hash was looked
        // up twice: once for the size, once for the list itself).
        let mut lists: Vec<Vec<u32>> = Vec::with_capacity(hashes.len());
        for &hash in hashes {
            match self.index.lookup_hash(hash)? {
                Some(ids) => lists.push(ids),
                // No file has this n-gram - empty result
                None => return Ok(BTreeSet::new()),
            }
        }

        // Start with the smallest list.
        lists.sort_by_key(Vec::len);
        let mut iter = lists.into_iter();
        let mut result = iter.next().unwrap();

        // Intersect with remaining lists using galloping
        for next_list in iter {
            result = galloping_intersect(&result, &next_list);
            if result.is_empty() {
                return Ok(BTreeSet::new());
            }
        }

        Ok(result.into_iter().collect())
    }

    /// Resolve a [`QueryAtom`] tree to a set of candidate file_ids.
    fn resolve_atom(&self, atom: &QueryAtom) -> Result<BTreeSet<u32>> {
        match atom {
            QueryAtom::Any => {
                // No constraints — all files are candidates.
                let n = self.index.file_count();
                Ok((0..n as u32).collect())
            }
            QueryAtom::And(hashes) => self.intersect_hashes(hashes),
            QueryAtom::Or(branches) => {
                let mut result = BTreeSet::new();
                for branch in branches {
                    let ids = self.resolve_atom(branch)?;
                    result.extend(ids);
                }
                Ok(result)
            }
        }
    }

    fn scan_all<F>(&self, literal: &str, load_content: &F) -> Result<Vec<SearchResult>>
    where
        F: Fn(u32, &std::path::Path) -> Option<Vec<u8>> + Sync,
    {
        let needle = literal.as_bytes();
        let verify = |file_id: u32| {
            let path = self.index.file_path(file_id)?.to_path_buf();
            let content = load_content(file_id, &path)?;
            // T7: Find ALL occurrences, not just first
            let offsets: Vec<usize> = memchr::memmem::find_iter(&content, needle).collect();
            if offsets.is_empty() {
                None
            } else {
                Some(SearchResult {
                    file_id,
                    path,
                    match_offsets: offsets,
                })
            }
        };
        let ids = (0..self.index.file_count() as u32).collect();
        Ok(verify_candidates(ids, verify))
    }

    fn verify_literal<F>(
        &self,
        literal: &str,
        candidates: BTreeSet<u32>,
        load_content: &F,
    ) -> Result<Vec<SearchResult>>
    where
        F: Fn(u32, &std::path::Path) -> Option<Vec<u8>> + Sync,
    {
        let needle = literal.as_bytes();
        let verify = |file_id: u32| {
            let path = self.index.file_path(file_id)?.to_path_buf();
            let content = load_content(file_id, &path)?;
            // T6: Non-overlapping matches (memmem::find_iter semantics).
            let offsets: Vec<usize> = memchr::memmem::find_iter(&content, needle).collect();
            if offsets.is_empty() {
                None
            } else {
                Some(SearchResult {
                    file_id,
                    path,
                    match_offsets: offsets,
                })
            }
        };
        Ok(verify_candidates(candidates.into_iter().collect(), verify))
    }
}

// ---------------------------------------------------------------------------
// Substring search (SIMD-accelerated memmem; used as ground truth in tests)
// ---------------------------------------------------------------------------

#[cfg(test)]
fn find_substring(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    memchr::memmem::find(haystack, needle)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::IndexBuilder;
    use std::path::Path;
    use tempfile::TempDir;

    struct TestRepo {
        files: Vec<(PathBuf, Vec<u8>)>,
    }

    impl TestRepo {
        fn new() -> Self {
            Self {
                files: vec![
                    (
                        PathBuf::from("main.rs"),
                        b"fn main() { println!(\"hello world\"); }".to_vec(),
                    ),
                    (
                        PathBuf::from("lib.rs"),
                        b"pub fn add(a: i32, b: i32) -> i32 { a + b }".to_vec(),
                    ),
                    (
                        PathBuf::from("util.rs"),
                        b"fn helper() { let x = 42; }".to_vec(),
                    ),
                    (
                        PathBuf::from("types.rs"),
                        b"struct Point { x: f64, y: f64 }".to_vec(),
                    ),
                ],
            }
        }

        fn build_index(&self, dir: &std::path::Path) -> Result<Index> {
            let mut builder = IndexBuilder::with_default_weights();
            for (path, content) in &self.files {
                builder.add_file(path, content)?;
            }
            builder.build(dir)
        }

        fn loader(&self) -> impl Fn(u32, &std::path::Path) -> Option<Vec<u8>> + Sync + '_ {
            |_file_id, path| {
                self.files
                    .iter()
                    .find(|(p, _)| p == path)
                    .map(|(_, c)| c.clone())
            }
        }
    }

    #[test]
    fn test_search_literal_found() -> Result<()> {
        let tmp = TempDir::new()?;
        let repo = TestRepo::new();
        let index = repo.build_index(tmp.path())?;
        let engine = QueryEngine::new(&index);

        let results = engine.search_literal("println", repo.loader())?;
        assert_eq!(results.len(), 1, "only main.rs contains println");
        assert_eq!(results[0].path, PathBuf::from("main.rs"));
        assert!(!results[0].match_offsets.is_empty());
        Ok(())
    }

    #[test]
    fn test_search_literal_not_found() -> Result<()> {
        let tmp = TempDir::new()?;
        let repo = TestRepo::new();
        let index = repo.build_index(tmp.path())?;
        let engine = QueryEngine::new(&index);

        let results = engine.search_literal("xyzzy_not_present_12345", repo.loader())?;
        assert!(results.is_empty());
        Ok(())
    }

    #[test]
    fn test_search_regex_found() -> Result<()> {
        let tmp = TempDir::new()?;
        let repo = TestRepo::new();
        let index = repo.build_index(tmp.path())?;
        let engine = QueryEngine::new(&index);

        // "struct [A-Z]" should match types.rs.
        let results = engine.search("struct [A-Z]", repo.loader())?;
        assert!(
            results.iter().any(|r| r.path == Path::new("types.rs")),
            "types.rs should match struct [A-Z]"
        );
        Ok(())
    }

    #[test]
    fn test_no_false_negatives_literal() -> Result<()> {
        let tmp = TempDir::new()?;
        let repo = TestRepo::new();
        let index = repo.build_index(tmp.path())?;
        let engine = QueryEngine::new(&index);

        // Verify every literal that exists in the test files is found.
        let needles = ["fn main", "pub fn", "println", "hello world", "i32", "f64"];
        for needle in &needles {
            let results = engine.search_literal(needle, repo.loader())?;
            // Brute-force ground truth.
            let expected: Vec<&PathBuf> = repo
                .files
                .iter()
                .filter(|(_, c)| find_substring(c, needle.as_bytes()).is_some())
                .map(|(p, _)| p)
                .collect();
            for p in &expected {
                assert!(
                    results.iter().any(|r| &r.path == *p),
                    "false negative: needle {:?} not found in {:?}",
                    needle,
                    p
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_unreadable_candidate_is_excluded_regex() -> Result<()> {
        // FP-1 regression: a candidate whose content cannot be loaded must
        // be excluded, not included without verification.
        let tmp = TempDir::new()?;
        let repo = TestRepo::new();
        let index = repo.build_index(tmp.path())?;
        let engine = QueryEngine::new(&index);

        let failing_loader = |_id: u32, _path: &std::path::Path| -> Option<Vec<u8>> { None };
        let results = engine.search("main", failing_loader)?;
        assert!(
            results.is_empty(),
            "unreadable candidates must not produce results"
        );
        Ok(())
    }

    #[test]
    fn test_unreadable_candidate_is_excluded_literal() -> Result<()> {
        let tmp = TempDir::new()?;
        let repo = TestRepo::new();
        let index = repo.build_index(tmp.path())?;
        let engine = QueryEngine::new(&index);

        let failing_loader = |_id: u32, _path: &std::path::Path| -> Option<Vec<u8>> { None };
        let results = engine.search_literal("println", failing_loader)?;
        assert!(
            results.is_empty(),
            "unreadable candidates must not produce results"
        );
        Ok(())
    }

    #[test]
    fn test_search_regex_no_false_negatives() -> Result<()> {
        let tmp = TempDir::new()?;
        let repo = TestRepo::new();
        let index = repo.build_index(tmp.path())?;
        let engine = QueryEngine::new(&index);

        let patterns = ["fn \\w+", "i32", "hello|world", "struct"];
        for pattern in &patterns {
            let re = Regex::new(pattern).unwrap();
            let results = engine.search(pattern, repo.loader())?;

            for (path, content) in &repo.files {
                let expected_match = re.find(content).is_some();
                let found = results.iter().any(|r| &r.path == path);
                if expected_match {
                    assert!(
                        found,
                        "false negative: pattern {:?} should match {:?}",
                        pattern, path
                    );
                }
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // MatchDetail / context line tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_offset_to_match_detail_single_line() {
        let content = b"fn main() {}";
        let detail = offset_to_match_detail(content, 3);
        assert_eq!(detail.line_number, 1);
        assert_eq!(detail.column, 4);
        assert_eq!(detail.line_content, "fn main() {}");
    }

    #[test]
    fn test_offset_to_match_detail_multi_line() {
        let content = b"line one\nline two\nline three";
        // "line two" starts at offset 9.
        let detail = offset_to_match_detail(content, 9);
        assert_eq!(detail.line_number, 2);
        assert_eq!(detail.column, 1);
        assert_eq!(detail.line_content, "line two");
    }

    #[test]
    fn test_offset_to_match_detail_last_line() {
        let content = b"aaa\nbbb\nccc";
        // "ccc" starts at offset 8.
        let detail = offset_to_match_detail(content, 8);
        assert_eq!(detail.line_number, 3);
        assert_eq!(detail.column, 1);
        assert_eq!(detail.line_content, "ccc");
    }

    #[test]
    fn test_extract_context_lines_middle() {
        let content = b"one\ntwo\nthree\nfour\nfive";
        // Line 3 = "three"; context 1 on each side → lines 2, 3, 4.
        let ctx = extract_context_lines(content, 3, 1, 1);
        assert_eq!(ctx.len(), 3);
        assert_eq!(ctx[0], (2, "two".to_string()));
        assert_eq!(ctx[1], (3, "three".to_string()));
        assert_eq!(ctx[2], (4, "four".to_string()));
    }

    #[test]
    fn test_extract_context_lines_boundary() {
        let content = b"one\ntwo\nthree";
        // Line 1, context 2 before (clamped), 1 after.
        let ctx = extract_context_lines(content, 1, 2, 1);
        assert_eq!(ctx.len(), 2);
        assert_eq!(ctx[0], (1, "one".to_string()));
        assert_eq!(ctx[1], (2, "two".to_string()));
    }

    #[test]
    fn test_extract_context_lines_first_line() {
        let content = b"alpha\nbeta\ngamma";
        let ctx = extract_context_lines(content, 1, 0, 0);
        assert_eq!(ctx.len(), 1);
        assert_eq!(ctx[0], (1, "alpha".to_string()));
    }

    #[test]
    fn test_galloping_intersect_basic() {
        let a = vec![1, 3, 5, 7, 9];
        let b = vec![2, 3, 4, 7, 8, 9, 10];
        let result = galloping_intersect(&a, &b);
        assert_eq!(result, vec![3, 7, 9]);
    }

    #[test]
    fn test_galloping_intersect_empty() {
        let a = vec![1, 2, 3];
        let b: Vec<u32> = vec![];
        let result = galloping_intersect(&a, &b);
        assert!(result.is_empty());
    }

    #[test]
    fn test_galloping_intersect_no_overlap() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        let result = galloping_intersect(&a, &b);
        assert!(result.is_empty());
    }
}
