//! End-to-end integration tests for `sparse-ngram-core`.
//!
//! Each test writes real files to a temporary directory, builds an on-disk
//! index, and exercises the full `IndexBuilder → Index::open → QueryEngine`
//! pipeline.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use sparse_ngram_core::index::{Index, IndexBuilder};
use sparse_ngram_core::query::QueryEngine;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helper: tiny in-memory "repo" that can build an on-disk index
// ---------------------------------------------------------------------------

struct Repo {
    files: HashMap<PathBuf, Vec<u8>>,
}

impl Repo {
    fn new() -> Self {
        Self {
            files: HashMap::new(),
        }
    }

    fn add(mut self, path: &str, content: &[u8]) -> Self {
        self.files.insert(PathBuf::from(path), content.to_vec());
        self
    }

    fn build_index(&self, dir: &Path) -> Result<Index> {
        let mut builder = IndexBuilder::with_default_weights();
        // Use a sorted iteration so file_ids are deterministic.
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
}

// ---------------------------------------------------------------------------
// 1. Literal search — hit
// ---------------------------------------------------------------------------

#[test]
fn literal_search_hit() -> Result<()> {
    let tmp = TempDir::new()?;
    let repo = Repo::new()
        .add("main.rs", b"fn main() { println!(\"hello world\"); }")
        .add("lib.rs", b"pub fn add(a: i32, b: i32) -> i32 { a + b }");

    let index = repo.build_index(tmp.path())?;
    let engine = QueryEngine::new(&index);

    let results = engine.search_literal("println", repo.loader())?;
    assert_eq!(results.len(), 1, "only main.rs contains println");
    assert_eq!(results[0].path, PathBuf::from("main.rs"));
    assert!(
        !results[0].match_offsets.is_empty(),
        "should report byte offsets"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 2. Literal search — miss
// ---------------------------------------------------------------------------

#[test]
fn literal_search_miss() -> Result<()> {
    let tmp = TempDir::new()?;
    let repo = Repo::new()
        .add("a.rs", b"fn foo() {}")
        .add("b.rs", b"fn bar() {}");

    let index = repo.build_index(tmp.path())?;
    let engine = QueryEngine::new(&index);

    let results = engine.search_literal("xyzzy_not_present_99999", repo.loader())?;
    assert!(results.is_empty(), "pattern must not be found");
    Ok(())
}

// ---------------------------------------------------------------------------
// 3. Regex search — hit
// ---------------------------------------------------------------------------

#[test]
fn regex_search_hit() -> Result<()> {
    let tmp = TempDir::new()?;
    let repo = Repo::new()
        .add("types.rs", b"struct Point { x: f64, y: f64 }")
        .add("funcs.rs", b"fn compute(val: f32) -> f32 { val * 2.0 }");

    let index = repo.build_index(tmp.path())?;
    let engine = QueryEngine::new(&index);

    let results = engine.search(r"struct [A-Z]\w+", repo.loader())?;
    assert!(
        results.iter().any(|r| r.path == Path::new("types.rs")),
        "types.rs should match struct [A-Z]\\w+"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 4. No false negatives — literal
// ---------------------------------------------------------------------------

#[test]
fn no_false_negatives_literal() -> Result<()> {
    let tmp = TempDir::new()?;
    let repo = Repo::new()
        .add("main.rs", b"fn main() { println!(\"hello world\"); }")
        .add("lib.rs", b"pub fn add(a: i32, b: i32) -> i32 { a + b }")
        .add("util.rs", b"fn helper() { let x = 42; }")
        .add("types.rs", b"struct Point { x: f64, y: f64 }");

    let index = repo.build_index(tmp.path())?;
    let engine = QueryEngine::new(&index);

    let needles = [
        "fn main",
        "pub fn",
        "println",
        "hello world",
        "i32",
        "f64",
        "struct",
    ];
    for needle in &needles {
        let results = engine.search_literal(needle, repo.loader())?;
        let result_paths: Vec<_> = results.iter().map(|r| &r.path).collect();

        for (path, content) in &repo.files {
            let present = content
                .windows(needle.len())
                .any(|w| w == needle.as_bytes());
            if present {
                assert!(
                    result_paths.contains(&path),
                    "false negative: needle {:?} not found in {:?}",
                    needle,
                    path
                );
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 5. No false negatives — regex
// ---------------------------------------------------------------------------

#[test]
fn no_false_negatives_regex() -> Result<()> {
    let tmp = TempDir::new()?;
    let repo = Repo::new()
        .add("main.rs", b"fn main() { println!(\"hello world\"); }")
        .add("lib.rs", b"pub fn add(a: i32, b: i32) -> i32 { a + b }")
        .add("util.rs", b"fn helper() { let x = 42; }")
        .add("types.rs", b"struct Point { x: f64, y: f64 }");

    let index = repo.build_index(tmp.path())?;
    let engine = QueryEngine::new(&index);

    let patterns = [r"fn \w+", r"i32", r"hello|world", r"struct"];
    for pattern in &patterns {
        let re = regex::bytes::Regex::new(pattern).unwrap();
        let results = engine.search(pattern, repo.loader())?;
        let result_paths: Vec<_> = results.iter().map(|r| &r.path).collect();

        for (path, content) in &repo.files {
            if re.is_match(content) {
                assert!(
                    result_paths.contains(&path),
                    "false negative: pattern {:?} should match {:?}",
                    pattern,
                    path
                );
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 6. Multi-file index — counts
// ---------------------------------------------------------------------------

#[test]
fn multi_file_index_counts() -> Result<()> {
    let tmp = TempDir::new()?;
    let mut repo = Repo::new();
    for i in 0..12 {
        let name = format!("file_{i}.rs");
        let content = format!("fn func_{i}() {{ let x = {i}; }}");
        repo = repo.add(Box::leak(name.into_boxed_str()), content.as_bytes());
    }

    let index = repo.build_index(tmp.path())?;
    assert_eq!(index.file_count(), 12, "should have 12 files indexed");
    assert!(index.ngram_count() > 0, "should have at least one n-gram");
    Ok(())
}

// ---------------------------------------------------------------------------
// 7. Rebuild consistency — ngram_count unchanged
// ---------------------------------------------------------------------------

#[test]
fn rebuild_consistency() -> Result<()> {
    let tmp1 = TempDir::new()?;
    let tmp2 = TempDir::new()?;
    let repo = Repo::new()
        .add("a.rs", b"fn alpha() { let a = 1; }")
        .add("b.rs", b"fn beta() { let b = 2; }");

    let index1 = repo.build_index(tmp1.path())?;
    let ngrams1 = index1.ngram_count();
    let files1 = index1.file_count();

    // Build again into a fresh directory — results must be identical.
    let index2 = repo.build_index(tmp2.path())?;
    assert_eq!(
        index2.ngram_count(),
        ngrams1,
        "ngram_count must be stable across rebuilds"
    );
    assert_eq!(
        index2.file_count(),
        files1,
        "file_count must be stable across rebuilds"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 8. Incremental add file — new file discoverable after rebuild
// ---------------------------------------------------------------------------

#[test]
fn incremental_add_file() -> Result<()> {
    let tmp = TempDir::new()?;
    let mut repo = Repo::new().add("existing.rs", b"fn existing() {}");
    let index = repo.build_index(tmp.path())?;

    // Search for a token that only exists in the new file — must not appear yet.
    let engine = QueryEngine::new(&index);
    let before = engine.search_literal("unique_new_token_xyz", repo.loader())?;
    assert!(before.is_empty());

    // Add new file and rebuild.
    repo = repo.add("new.rs", b"fn unique_new_token_xyz() {}");
    let tmp2 = TempDir::new()?;
    let index2 = repo.build_index(tmp2.path())?;
    let engine2 = QueryEngine::new(&index2);

    let after = engine2.search_literal("unique_new_token_xyz", repo.loader())?;
    assert_eq!(
        after.len(),
        1,
        "new file must be discoverable after rebuild"
    );
    assert_eq!(after[0].path, PathBuf::from("new.rs"));
    Ok(())
}

// ---------------------------------------------------------------------------
// 9. Empty index — searches return empty
// ---------------------------------------------------------------------------

#[test]
fn empty_index_returns_empty() -> Result<()> {
    let tmp = TempDir::new()?;
    let repo = Repo::new(); // no files
    let index = repo.build_index(tmp.path())?;

    assert_eq!(index.file_count(), 0);
    let engine = QueryEngine::new(&index);
    let results = engine.search_literal("anything", repo.loader())?;
    assert!(results.is_empty());
    Ok(())
}

// ---------------------------------------------------------------------------
// 10. Unicode / multi-byte content
// ---------------------------------------------------------------------------

#[test]
fn unicode_content_indexed_and_found() -> Result<()> {
    let tmp = TempDir::new()?;
    let repo = Repo::new()
        .add(
            "unicode.rs",
            "// 你好世界\nfn greet() { println!(\"こんにちは\"); }".as_bytes(),
        )
        .add("ascii.rs", b"fn noop() {}");

    let index = repo.build_index(tmp.path())?;
    let engine = QueryEngine::new(&index);

    // Search for ASCII token that only exists in the unicode file.
    let results = engine.search_literal("greet", repo.loader())?;
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].path, PathBuf::from("unicode.rs"));
    Ok(())
}

// ---------------------------------------------------------------------------
// 11. Multiple matches in a single file
// ---------------------------------------------------------------------------

#[test]
fn multiple_matches_in_file() -> Result<()> {
    let tmp = TempDir::new()?;
    let repo = Repo::new().add("repeat.rs", b"let x = 1; let y = 1; let z = 1;");

    let index = repo.build_index(tmp.path())?;
    let engine = QueryEngine::new(&index);

    let results = engine.search_literal("let", repo.loader())?;
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].match_offsets.len(),
        3,
        "should find 3 occurrences of 'let'"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 12. Search after Index::open (disk round-trip)
// ---------------------------------------------------------------------------

#[test]
fn search_after_disk_roundtrip() -> Result<()> {
    let tmp = TempDir::new()?;
    let repo = Repo::new().add("disk.rs", b"fn disk_roundtrip_marker() {}");

    // Build and drop.
    {
        repo.build_index(tmp.path())?;
    }

    // Re-open from disk.
    let index = Index::open(tmp.path())?;
    let engine = QueryEngine::new(&index);

    let results = engine.search_literal("disk_roundtrip_marker", repo.loader())?;
    assert_eq!(
        results.len(),
        1,
        "token must be found after disk round-trip"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Chinese / CJK search tests
// ---------------------------------------------------------------------------

#[test]
fn chinese_literal_search_hit() -> Result<()> {
    let tmp = TempDir::new()?;
    let repo = Repo::new()
        .add("chinese.rs", "// 你好世界\nfn greet() {}".as_bytes())
        .add("other.rs", b"fn other() {}");

    let index = repo.build_index(tmp.path())?;
    let engine = QueryEngine::new(&index);

    let results = engine.search_literal("你好世界", repo.loader())?;
    assert_eq!(results.len(), 1, "only chinese.rs contains 你好世界");
    assert_eq!(results[0].path, PathBuf::from("chinese.rs"));
    assert!(!results[0].match_offsets.is_empty());
    Ok(())
}

#[test]
fn chinese_literal_search_miss() -> Result<()> {
    let tmp = TempDir::new()?;
    let repo = Repo::new()
        .add("a.rs", "// 你好世界".as_bytes())
        .add("b.rs", b"fn foo() {}");

    let index = repo.build_index(tmp.path())?;
    let engine = QueryEngine::new(&index);

    let results = engine.search_literal("再见世界", repo.loader())?;
    assert!(results.is_empty(), "再见世界 is not present");
    Ok(())
}

#[test]
fn chinese_literal_no_false_negatives() -> Result<()> {
    let content = "// 你好世界\nfn hello() {}\n// 函数 and 变量\n".as_bytes();
    let tmp = TempDir::new()?;
    let repo = Repo::new()
        .add("zh.rs", content)
        .add("ascii.rs", b"fn noop() {}");

    let index = repo.build_index(tmp.path())?;
    let engine = QueryEngine::new(&index);

    for needle in &["你好", "世界", "函数", "变量"] {
        let results = engine.search_literal(needle, repo.loader())?;
        assert!(
            results.iter().any(|r| r.path == Path::new("zh.rs")),
            "false negative: Chinese needle {:?} not found in zh.rs",
            needle
        );
    }
    Ok(())
}

#[test]
fn chinese_regex_search() -> Result<()> {
    let tmp = TempDir::new()?;
    let repo = Repo::new()
        .add("zh.rs", "// 你好世界\nfn greet() {}".as_bytes())
        .add("other.rs", b"fn other() {}");

    let index = repo.build_index(tmp.path())?;
    let engine = QueryEngine::new(&index);

    let results = engine.search("你好.*世界", repo.loader())?;
    assert!(
        results.iter().any(|r| r.path == Path::new("zh.rs")),
        "regex 你好.*世界 should match zh.rs"
    );
    Ok(())
}

#[test]
fn mixed_chinese_ascii_search() -> Result<()> {
    let content = "// 这是注释\nfn hello() {}".as_bytes();
    let tmp = TempDir::new()?;
    let repo = Repo::new()
        .add("mixed.rs", content)
        .add("ascii.rs", b"fn world() {}");

    let index = repo.build_index(tmp.path())?;
    let engine = QueryEngine::new(&index);

    // Search Chinese.
    let results = engine.search_literal("这是注释", repo.loader())?;
    assert!(
        results.iter().any(|r| r.path == Path::new("mixed.rs")),
        "Chinese literal should match mixed.rs"
    );

    // Search ASCII in the same file.
    let results = engine.search_literal("hello", repo.loader())?;
    assert!(
        results.iter().any(|r| r.path == Path::new("mixed.rs")),
        "ASCII literal should also match mixed.rs"
    );
    Ok(())
}

#[test]
fn chinese_multiple_matches_in_file() -> Result<()> {
    let content = "你好 世界\n你好 朋友\n你好 大家".as_bytes();
    let tmp = TempDir::new()?;
    let repo = Repo::new().add("repeat.rs", content);

    let index = repo.build_index(tmp.path())?;
    let engine = QueryEngine::new(&index);

    let results = engine.search_literal("你好", repo.loader())?;
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].match_offsets.len(),
        3,
        "should find 3 occurrences of 你好"
    );
    Ok(())
}

#[test]
fn emoji_content_search() -> Result<()> {
    // Emoji are 4-byte UTF-8 sequences.
    let content = "fn emoji() { /* 🦀 Rust */ }".as_bytes();
    let tmp = TempDir::new()?;
    let repo = Repo::new()
        .add("emoji.rs", content)
        .add("plain.rs", b"fn plain() {}");

    let index = repo.build_index(tmp.path())?;
    let engine = QueryEngine::new(&index);

    // Search for the ASCII part present only in emoji.rs.
    let results = engine.search_literal("emoji", repo.loader())?;
    assert!(
        results.iter().any(|r| r.path == Path::new("emoji.rs")),
        "emoji.rs should be findable"
    );
    Ok(())
}

#[test]
fn chinese_across_multiple_files() -> Result<()> {
    let tmp = TempDir::new()?;
    let repo = Repo::new()
        .add("file_a.rs", "// 苹果 apple".as_bytes())
        .add("file_b.rs", "// 橙子 orange".as_bytes())
        .add("file_c.rs", b"// just ASCII");

    let index = repo.build_index(tmp.path())?;
    let engine = QueryEngine::new(&index);

    let results = engine.search_literal("苹果", repo.loader())?;
    assert_eq!(results.len(), 1, "only file_a.rs contains 苹果");
    assert_eq!(results[0].path, PathBuf::from("file_a.rs"));

    let results = engine.search_literal("橙子", repo.loader())?;
    assert_eq!(results.len(), 1, "only file_b.rs contains 橙子");
    assert_eq!(results[0].path, PathBuf::from("file_b.rs"));
    Ok(())
}

// ---------------------------------------------------------------------------
// MatchDetail / context line tests (integration)
// ---------------------------------------------------------------------------

#[test]
fn offset_to_match_detail_integration() -> Result<()> {
    use sparse_ngram_core::query::offset_to_match_detail;

    let content = b"fn foo() {}\nfn bar() {}\nfn baz() {}";
    // "bar" starts at offset 15 (after "fn foo() {}\n" = 12 bytes, then "fn " = 3).
    let offset = 15;
    let detail = offset_to_match_detail(content, offset);
    assert_eq!(detail.line_number, 2);
    assert_eq!(detail.line_content, "fn bar() {}");
    Ok(())
}

#[test]
fn extract_context_lines_integration() -> Result<()> {
    use sparse_ngram_core::query::extract_context_lines;

    let content = b"line1\nline2\nline3\nline4\nline5";
    let ctx = extract_context_lines(content, 3, 1, 1);
    let lines: Vec<_> = ctx.iter().map(|(_, s)| s.as_str()).collect();
    assert_eq!(lines, vec!["line2", "line3", "line4"]);
    Ok(())
}
