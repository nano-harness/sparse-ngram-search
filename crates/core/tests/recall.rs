//! Recall/precision regression tests: index-backed search vs. brute force.
//!
//! These tests build an index over a deterministic pseudo-random corpus and
//! compare every query result against a brute-force ground truth
//! (`memchr::memmem` for literals, `regex::bytes::Regex` for patterns).
//! The contract under test is: **zero false negatives and zero false
//! positives** on the indexed corpus.

use std::collections::BTreeSet;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use regex::bytes::Regex;
use sparse_ngram_core::index::{Index, IndexBuilder};
use sparse_ngram_core::query::QueryEngine;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Deterministic PRNG (xorshift64*) — no external deps, stable across runs
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }
}

// ---------------------------------------------------------------------------
// Corpus generation
// ---------------------------------------------------------------------------

/// Vocabulary mixing plain ASCII tokens, code-like fragments, CJK text and
/// emoji, so n-gram weight boundaries get exercised across scripts.
const VOCAB: &[&str] = &[
    "alpha",
    "beta",
    "gamma",
    "delta",
    "needle123",
    "fn",
    "struct",
    "impl",
    "let",
    "match",
    "return",
    "error",
    "warning",
    "ConnectionPool",
    "parse_header",
    "TODO",
    "unsafe",
    "mutex",
    "buffer",
    "stream",
    "搜索",
    "索引",
    "召回率",
    "准确率",
    "🎉",
    "🚀",
    "x",
    "y",
    "=>",
    "::",
    "(",
    ")",
    "{",
    "}",
    ";",
    ".",
    "/",
    "_",
    "0",
    "42",
    "foo",
    "bar",
    "baz",
];

const FRAGMENTS: &[&str] = &[
    "fn parse_header(buf: &[u8]) -> Result<Header>",
    "let pool = ConnectionPool::new(config);",
    "// TODO: handle the edge case",
    "搜索 索引 召回率 准确率",
    "unsafe { ptr::read(src) }",
    "alpha beta gamma delta",
];

fn generate_corpus(seed: u64, file_count: usize) -> Vec<(PathBuf, Vec<u8>)> {
    let mut rng = Rng::new(seed);
    let mut files = Vec::with_capacity(file_count);

    for i in 0..file_count {
        let mut content = String::new();
        let lines = 5 + rng.below(40);
        for _ in 0..lines {
            // ~1/4 of lines embed a shared fragment so posting-list
            // intersections actually have work to do.
            if rng.below(4) == 0 {
                #[allow(clippy::explicit_auto_deref)]
                content.push_str(*rng.pick(FRAGMENTS));
            } else {
                let words = 1 + rng.below(10);
                for _ in 0..words {
                    #[allow(clippy::explicit_auto_deref)]
                    content.push_str(*rng.pick(VOCAB));
                    content.push(' ');
                }
            }
            content.push('\n');
        }
        files.push((
            PathBuf::from(format!("src/file_{i:03}.rs")),
            content.into_bytes(),
        ));
    }

    // Edge-case files.
    files.push((PathBuf::from("empty.rs"), Vec::new()));
    files.push((PathBuf::from("one_byte.rs"), b"a".to_vec()));
    files.push((
        PathBuf::from("no_newline.rs"),
        b"tail without newline".to_vec(),
    ));
    files.push((
        PathBuf::from("unicode_only.rs"),
        "搜索\n索引\n召回率\n".as_bytes().to_vec(),
    ));
    // CRLF line endings: matches may span or abut '\r' bytes.
    files.push((
        PathBuf::from("crlf.rs"),
        b"fn parse_header(buf: &[u8]) -> Result<Header>\r\nalpha beta\r\ngamma delta\r\n".to_vec(),
    ));
    // A file whose tail is one long line with no newline at all.
    files.push((
        PathBuf::from("long_line.rs"),
        b"alpha beta gamma delta needle123 ConnectionPool parse_header TODO unsafe mutex".to_vec(),
    ));
    files
}

// ---------------------------------------------------------------------------
// In-memory repo harness
// ---------------------------------------------------------------------------

struct Repo {
    files: HashMap<PathBuf, Vec<u8>>,
}

impl Repo {
    fn from(files: Vec<(PathBuf, Vec<u8>)>) -> Self {
        Self {
            files: files.into_iter().collect(),
        }
    }

    fn build_index(&self, dir: &Path) -> Result<Index> {
        let mut builder = IndexBuilder::with_default_weights();
        let mut sorted: Vec<_> = self.files.iter().collect();
        sorted.sort_by_key(|(p, _)| (*p).clone());
        for (path, content) in sorted {
            builder.add_file(path, content)?;
        }
        builder.build(dir)
    }

    fn loader(&self) -> impl Fn(u32, &Path) -> Option<Vec<u8>> + Sync + '_ {
        |_id, path| self.files.get(path).cloned()
    }

    fn literal_ground_truth(&self, needle: &str) -> BTreeSet<PathBuf> {
        self.files
            .iter()
            .filter(|(_, c)| memchr::memmem::find(c, needle.as_bytes()).is_some())
            .map(|(p, _)| p.clone())
            .collect()
    }

    fn regex_ground_truth(&self, pattern: &str) -> BTreeSet<PathBuf> {
        let re = Regex::new(pattern).unwrap();
        self.files
            .iter()
            .filter(|(_, c)| re.find(c).is_some())
            .map(|(p, _)| p.clone())
            .collect()
    }
}

fn result_paths(results: &[sparse_ngram_core::query::SearchResult]) -> BTreeSet<PathBuf> {
    results.iter().map(|r| r.path.clone()).collect()
}

// ---------------------------------------------------------------------------
// Query sets
// ---------------------------------------------------------------------------

/// Literals sampled from the corpus vocabulary plus absent strings and
/// single-character needles (which force the scan-all fallback).
fn literal_queries(corpus: &[(PathBuf, Vec<u8>)]) -> Vec<String> {
    let mut rng = Rng::new(0xC0FFEE);
    let mut queries: Vec<String> = Vec::new();

    // Vocabulary tokens (present by construction).
    for tok in VOCAB.iter().filter(|t| !t.is_empty()) {
        queries.push(tok.to_string());
    }
    // Fragment substrings.
    for frag in FRAGMENTS {
        queries.push(frag.to_string());
        if frag.len() > 8 {
            queries.push(frag.chars().take(8).collect());
        }
    }
    // Absent strings.
    queries.push("zzz_absent_99999".to_string());
    queries.push("qyqyqy_never_indexed".to_string());
    // Single characters → scan_all path.
    queries.push("a".to_string());
    queries.push("搜".to_string());
    // Random adjacent token pairs sampled from real file content.
    for _ in 0..40 {
        let (path, content) = rng.pick(corpus);
        let _ = path;
        let text = String::from_utf8_lossy(content);
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() >= 2 {
            let i = rng.below(words.len() - 1);
            queries.push(format!("{} {}", words[i], words[i + 1]));
        }
    }
    queries.sort();
    queries.dedup();
    queries
}

/// Regex templates, heavy on the historically fragile corners:
/// alternation with an unconstrained branch, optional groups, case
/// folding, nested alternation, and dot-star spans.
fn regex_queries() -> Vec<&'static str> {
    vec![
        // Plain literals and classes.
        "needle123",
        r"fn \w+",
        r"struct\s+\w+",
        r"ConnectionPool",
        // Alternation of constrained literals.
        "alpha|beta",
        "foo|bar|baz",
        "(alpha|beta) gamma",
        // FN-1 regression: alternation with an unconstrained branch must
        // degrade to a full scan, not silently drop the branch.
        "zzz_absent_99999|.*",
        "needle123|x?",
        "alpha|.",
        "(foo|x?)",
        r"error|warn\d*",
        // Optional groups.
        "colou?r",
        "err(or)?",
        "beta?",
        // Dot-star spans.
        "alpha.*delta",
        "fn .*\\(",
        // Case folding.
        "(?i)connectionpool",
        "(?i)ALPHA",
        "(?i)todo",
        // Anchors and looks.
        "^fn ",
        r"\bunsafe\b",
        // Matches everything, including the empty file.
        ".*",
        "a*",
        // Absent matches.
        "zzz_absent_99999|qyqyqy_never_indexed",
        r"\d{5,}",
    ]
}

// ---------------------------------------------------------------------------
// The differential test
// ---------------------------------------------------------------------------

#[test]
fn recall_and_precision_match_brute_force() -> Result<()> {
    let corpus = generate_corpus(0x5EED, 200);
    let repo = Repo::from(corpus.clone());
    let tmp = TempDir::new()?;
    let index = repo.build_index(tmp.path())?;
    let engine = QueryEngine::new(&index);

    let mut checked = 0usize;

    for needle in literal_queries(&corpus) {
        let results = engine.search_literal(&needle, repo.loader())?;
        let got = result_paths(&results);
        let want = repo.literal_ground_truth(&needle);
        assert_eq!(
            got,
            want,
            "literal {needle:?}: result set differs from brute force \
             (missing: {:?}, extra: {:?})",
            want.difference(&got).collect::<Vec<_>>(),
            got.difference(&want).collect::<Vec<_>>()
        );
        checked += 1;
    }

    for pattern in regex_queries() {
        let results = engine.search(pattern, repo.loader())?;
        let got = result_paths(&results);
        let want = repo.regex_ground_truth(pattern);
        assert_eq!(
            got,
            want,
            "regex {pattern:?}: result set differs from brute force \
             (missing: {:?}, extra: {:?})",
            want.difference(&got).collect::<Vec<_>>(),
            got.difference(&want).collect::<Vec<_>>()
        );
        checked += 1;
    }

    assert!(checked > 100, "sanity: expected a substantial query set");
    Ok(())
}

// ---------------------------------------------------------------------------
// Fuzz: random substring literals (the strongest n-gram covering stress)
// ---------------------------------------------------------------------------

/// Sample `count` random substrings of actual file content at arbitrary byte
/// offsets (snapped to char boundaries so they are valid `&str`).  Unlike
/// token-based queries, these routinely start and end in the middle of
/// tokens, which is where the sparse n-gram covering invariant is easiest
/// to break.
fn random_substring_queries(corpus: &[(PathBuf, Vec<u8>)], seed: u64, count: usize) -> Vec<String> {
    let mut rng = Rng::new(seed);
    let mut queries = Vec::new();
    for _ in 0..count {
        let (_, content) = rng.pick(corpus);
        let Ok(text) = std::str::from_utf8(content) else {
            continue;
        };
        if text.is_empty() {
            continue;
        }
        let len = text.len();
        let mut start = rng.below(len);
        while !text.is_char_boundary(start) {
            start += 1;
            if start >= len {
                break;
            }
        }
        let max_len = (len - start).min(48);
        if max_len == 0 {
            continue;
        }
        let mut end = start + 1 + rng.below(max_len);
        while end < len && !text.is_char_boundary(end) {
            end += 1;
        }
        let q = &text[start..end];
        if !q.is_empty() {
            queries.push(q.to_string());
        }
    }
    queries.sort();
    queries.dedup();
    queries
}

#[test]
fn fuzz_random_substring_literals_match_brute_force() -> Result<()> {
    let mut checked = 0usize;

    for seed in [1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10] {
        let corpus = generate_corpus(seed * 0x5EED, 150);
        let repo = Repo::from(corpus.clone());
        let tmp = TempDir::new()?;
        let index = repo.build_index(tmp.path())?;
        let engine = QueryEngine::new(&index);

        for needle in random_substring_queries(&corpus, seed * 0xFACE, 120) {
            let results = engine.search_literal(&needle, repo.loader())?;
            let got = result_paths(&results);
            let want = repo.literal_ground_truth(&needle);
            assert_eq!(
                got,
                want,
                "literal {needle:?}: result set differs from brute force \
                 (missing: {:?}, extra: {:?})",
                want.difference(&got).collect::<Vec<_>>(),
                got.difference(&want).collect::<Vec<_>>()
            );

            // Precision of reported offsets: they must equal brute-force
            // non-overlapping match positions exactly.
            for r in &results {
                let content = &repo.files[&r.path];
                let want_offsets: Vec<usize> =
                    memchr::memmem::find_iter(content, needle.as_bytes()).collect();
                assert_eq!(
                    r.match_offsets, want_offsets,
                    "literal {needle:?} in {:?}: offsets differ",
                    r.path
                );
            }
            checked += 1;
        }
    }

    assert!(checked > 200, "sanity: expected a substantial query set");
    Ok(())
}

// ---------------------------------------------------------------------------
// Fuzz: extended regex feature matrix across multiple corpora
// ---------------------------------------------------------------------------

/// Regex features beyond the base set: character classes, bounded
/// repetitions, group repetitions, multiline anchors, nested alternation,
/// and case folding over longer words.
fn extended_regex_queries() -> Vec<&'static str> {
    vec![
        // Character classes.
        r"[a-c]lpha",
        r"needle[0-9]+",
        r"[A-Z][a-z]+Pool",
        r"[xyz] ",
        r"[^\n]*needle123",
        // Bounded and group repetitions.
        r"\d{2,3}",
        r"(ab)+",
        r"(foo )+bar",
        r"x{3}",
        r"(?:alpha|beta){1,2}",
        // Multiline anchors.
        r"(?m)^fn ",
        r"(?m)^alpha",
        r"(?m)delta$",
        r"(?m)^TODO",
        // Nested alternation / groups.
        r"((alpha|beta) (gamma|delta))",
        r"fn (parse|check)_\w+",
        r"(unsafe|mutex) \{?",
        // Case folding.
        "(?i)WARNING",
        "(?i)unsafe",
        "(?i)needle\\d+",
        "(?i)(alpha|GAMMA)",
        // Dot without (?s) never crosses '\n'.
        r"alpha.*gamma",
        r"TODO: .*case",
        // Escaped metacharacters.
        r"ptr::read\(src\)",
        r"ConnectionPool::new",
        r"\{ src \}",
        // Word boundaries around CJK-adjacent tokens.
        r"\bfn\b",
        r"\b42\b",
        // Alternation where one branch is a strict prefix of another.
        r"error|error code",
        r"let|let pool",
    ]
}

#[test]
fn fuzz_extended_regex_matches_brute_force() -> Result<()> {
    let mut checked = 0usize;

    for seed in [1u64, 2, 3, 4, 5] {
        let corpus = generate_corpus(seed * 0xBEEF, 150);
        let repo = Repo::from(corpus.clone());
        let tmp = TempDir::new()?;
        let index = repo.build_index(tmp.path())?;
        let engine = QueryEngine::new(&index);

        for pattern in extended_regex_queries() {
            let results = engine.search(pattern, repo.loader())?;
            let got = result_paths(&results);
            let want = repo.regex_ground_truth(pattern);
            assert_eq!(
                got,
                want,
                "regex {pattern:?}: result set differs from brute force \
                 (missing: {:?}, extra: {:?})",
                want.difference(&got).collect::<Vec<_>>(),
                got.difference(&want).collect::<Vec<_>>()
            );
            checked += 1;
        }
    }

    assert!(checked > 50, "sanity: expected a substantial query set");
    Ok(())
}
