//! Performance benchmark suite for sparse-ngram-search.
//!
//! Measures three key metrics:
//!   1. **Indexing time**  — time to build an index from N files of K bytes each.
//!   2. **Search time**    — time to execute a literal / regex query against the index.
//!   3. **Index size**     — bytes consumed on disk by the generated index files.
//!
//! Run with:
//!   ```
//!   cargo run -p sparse-ngram-bench --release
//!   ```
//!
//! The results are printed as a Markdown-compatible table to stdout so they can
//! be copy-pasted as a baseline reference in documentation or CI artefacts.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::Result;
use sparse_ngram_core::index::Index;
use sparse_ngram_core::index::IndexBuilder;
use sparse_ngram_core::query::QueryEngine;

// ---------------------------------------------------------------------------
// Benchmark configuration
// ---------------------------------------------------------------------------

/// A single benchmark scenario: a corpus size and a set of query patterns.
struct Scenario {
    /// Human-readable label, e.g. "100 files × 4 KiB".
    label: &'static str,
    /// Number of synthetic files to index.
    file_count: usize,
    /// Approximate byte size of each file.
    file_size: usize,
    /// Number of times to repeat each measurement (for stable averages).
    iterations: u32,
}

const SCENARIOS: &[Scenario] = &[
    Scenario {
        label: "100 files × 1 KiB",
        file_count: 100,
        file_size: 1_024,
        iterations: 10,
    },
    Scenario {
        label: "100 files × 4 KiB",
        file_count: 100,
        file_size: 4_096,
        iterations: 10,
    },
    Scenario {
        label: "1000 files × 4 KiB",
        file_count: 1_000,
        file_size: 4_096,
        iterations: 5,
    },
    Scenario {
        label: "1000 files × 16 KiB",
        file_count: 1_000,
        file_size: 16_384,
        iterations: 3,
    },
    Scenario {
        label: "5000 files × 4 KiB",
        file_count: 5_000,
        file_size: 4_096,
        iterations: 3,
    },
];

/// Literal-search queries (must exist in the synthetic corpus).
const LITERAL_QUERIES: &[&str] = &["fn func_", "let x0 =", "return value", "impl Trait"];

/// Regex-search queries.
const REGEX_QUERIES: &[&str] = &[
    r"fn \w+_\d+",
    r"let x\d+ =",
    r"file_\d+",
    // Long case-insensitive literal: exercises the MAX_LITERALS prefix cap in
    // the decomposer (2^11 case combinations → 256 indexed 8-byte prefixes
    // instead of a full scan).  `zztransform9999` is absent from the
    // corpus, so the after-fix path resolves with zero candidates.
    "(?i)zztransform9999",
];

// ---------------------------------------------------------------------------
// Corpus generation
// ---------------------------------------------------------------------------

/// Generate `n` synthetic Rust-like source files of approximately `size` bytes each.
///
/// The content is deterministic and varied enough to produce realistic n-gram
/// distributions (different identifiers, numbers, structures per file).
fn generate_corpus(n: usize, size: usize) -> Vec<(PathBuf, Vec<u8>)> {
    (0..n)
        .map(|i| {
            let name = PathBuf::from(format!("src/file_{i:04}.rs"));
            let mut content = Vec::with_capacity(size + 128);

            // File header
            content.extend_from_slice(
                format!(
                    "// Auto-generated file {i}\n\
                     use std::collections::HashMap;\n\n\
                     pub struct Module{i} {{\n    \
                     data: Vec<u8>,\n    \
                     index: HashMap<String, usize>,\n}}\n\n\
                     impl Module{i} {{\n    \
                     pub fn new() -> Self {{\n        \
                     Self {{ data: Vec::new(), index: HashMap::new() }}\n    }}\n\n"
                )
                .as_bytes(),
            );

            // Filler: a series of small functions with varied patterns
            let line_template = [
                "fn func_{j}(x: u32, y: u32) -> u32 {{ x.wrapping_add(y).wrapping_mul({i}) }}\n",
                "fn transform_{j}(input: &[u8]) -> Vec<u8> {{ input.iter().map(|&b| b ^ {j}).collect() }}\n",
                "fn check_{j}(value: u64) -> bool {{ value % {j_plus} == 0 }}\n",
                "/// Compute result for index {j}\npub fn compute_{j}() -> i64 {{ {j} * {i} + 1 }}\n",
            ];

            let mut j = 0usize;
            while content.len() < size {
                let tmpl = line_template[j % line_template.len()];
                let line = tmpl
                    .replace("{j}", &j.to_string())
                    .replace("{i}", &i.to_string())
                    .replace("{j_plus}", &(j + 1).to_string());
                content.extend_from_slice(line.as_bytes());
                // Add a local variable to improve n-gram variety
                content.extend_from_slice(
                    format!(
                        "fn local_{j}() {{\n    let x{j} = {j};\n    let result = x{j} + {i};\n    \
                         return value + result;\n}}\n"
                    )
                    .as_bytes(),
                );
                j += 1;
            }

            content.extend_from_slice(b"} // end impl\n");
            (name, content)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Measurement helpers
// ---------------------------------------------------------------------------

/// Run `f` exactly `iters` times and return the total elapsed `Duration`.
fn measure<F: FnMut()>(iters: u32, mut f: F) -> Duration {
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed()
}

/// Return the total byte size of all files under `dir`.
fn dir_size(dir: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                total += entry.metadata().map(|m| m.len()).unwrap_or(0);
            } else if path.is_dir() {
                total += dir_size(&path);
            }
        }
    }
    total
}

/// Format a `Duration` as a human-readable string (µs / ms / s).
fn fmt_duration(d: Duration) -> String {
    let ns = d.as_nanos();
    if ns < 1_000_000 {
        format!("{:.1} µs", ns as f64 / 1_000.0)
    } else if ns < 1_000_000_000 {
        format!("{:.2} ms", ns as f64 / 1_000_000.0)
    } else {
        format!("{:.3} s", d.as_secs_f64())
    }
}

/// Format bytes as KiB / MiB / GiB.
fn fmt_bytes(b: u64) -> String {
    if b < 1024 {
        format!("{b} B")
    } else if b < 1024 * 1024 {
        format!("{:.1} KiB", b as f64 / 1024.0)
    } else if b < 1024 * 1024 * 1024 {
        format!("{:.2} MiB", b as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.3} GiB", b as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

// ---------------------------------------------------------------------------
// Result records
// ---------------------------------------------------------------------------

struct IndexingResult {
    scenario: &'static str,
    file_count: usize,
    corpus_bytes: u64,
    avg_index_time: Duration,
    index_bytes: u64,
    ngram_count: u64,
    /// index_bytes / corpus_bytes ratio expressed as a percentage.
    size_ratio_pct: f64,
}

struct SearchResult {
    scenario: &'static str,
    query: &'static str,
    kind: &'static str,
    avg_search_time: Duration,
    result_count: usize,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    println!("sparse-ngram-search — Performance Benchmark");
    println!("============================================\n");

    let mut indexing_results: Vec<IndexingResult> = Vec::new();
    let mut search_results: Vec<SearchResult> = Vec::new();

    for scenario in SCENARIOS {
        eprintln!("[bench] scenario: {}", scenario.label);

        // ------------------------------------------------------------------
        // 1. Corpus
        // ------------------------------------------------------------------
        let corpus = generate_corpus(scenario.file_count, scenario.file_size);
        let corpus_bytes: u64 = corpus.iter().map(|(_, c)| c.len() as u64).sum();

        // ------------------------------------------------------------------
        // 2. Indexing time
        // ------------------------------------------------------------------
        let index_dir = tempfile::TempDir::new()?;

        let mut last_index: Option<Index> = None;
        let index_elapsed = {
            let start = std::time::Instant::now();
            for iter in 0..scenario.iterations {
                eprintln!(
                    "[bench]   indexing iter {}/{}...",
                    iter + 1,
                    scenario.iterations
                );
                let mut b = IndexBuilder::with_default_weights();
                for (path, content) in &corpus {
                    b.add_file(path, content).unwrap();
                }
                let idx = b.build(index_dir.path()).unwrap();
                last_index = Some(idx);
            }
            start.elapsed()
        };

        let avg_index_time = index_elapsed / scenario.iterations;

        // Measure the size of the last-written index on disk.
        let index_bytes = dir_size(index_dir.path());

        // Reuse the last-built index for searching.
        let index = last_index.unwrap();
        let ngram_count = index.ngram_count();

        let size_ratio_pct = if corpus_bytes > 0 {
            index_bytes as f64 / corpus_bytes as f64 * 100.0
        } else {
            0.0
        };

        indexing_results.push(IndexingResult {
            scenario: scenario.label,
            file_count: scenario.file_count,
            corpus_bytes,
            avg_index_time,
            index_bytes,
            ngram_count,
            size_ratio_pct,
        });

        // ------------------------------------------------------------------
        // 3. Search time — literal
        // ------------------------------------------------------------------
        let engine = QueryEngine::new(&index);

        // Build a path→content lookup for verification.
        let path_map: std::collections::HashMap<PathBuf, &[u8]> = corpus
            .iter()
            .map(|(p, c)| (p.clone(), c.as_slice()))
            .collect();

        for &query in LITERAL_QUERIES {
            let query_str = query;
            let search_elapsed = measure(scenario.iterations, || {
                let _ = engine
                    .search_literal(query_str, |_id, path| {
                        path_map.get(path).map(|b| b.to_vec())
                    })
                    .unwrap();
            });
            let avg_search_time = search_elapsed / scenario.iterations;

            // Run once to get result count.
            let results =
                engine.search_literal(query, |_id, path| path_map.get(path).map(|b| b.to_vec()))?;

            search_results.push(SearchResult {
                scenario: scenario.label,
                query,
                kind: "literal",
                avg_search_time,
                result_count: results.len(),
            });
        }

        // ------------------------------------------------------------------
        // 4. Search time — regex
        // ------------------------------------------------------------------
        for &query in REGEX_QUERIES {
            let query_str = query;
            let search_elapsed = measure(scenario.iterations, || {
                let _ = engine
                    .search(query_str, |_id, path| {
                        path_map.get(path).map(|b| b.to_vec())
                    })
                    .unwrap();
            });
            let avg_search_time = search_elapsed / scenario.iterations;

            let results =
                engine.search(query, |_id, path| path_map.get(path).map(|b| b.to_vec()))?;

            search_results.push(SearchResult {
                scenario: scenario.label,
                query,
                kind: "regex",
                avg_search_time,
                result_count: results.len(),
            });
        }
    }

    // -----------------------------------------------------------------------
    // Print results
    // -----------------------------------------------------------------------
    print_indexing_table(&indexing_results);
    println!();
    print_search_table(&search_results);

    Ok(())
}

// ---------------------------------------------------------------------------
// Table rendering
// ---------------------------------------------------------------------------

fn print_indexing_table(rows: &[IndexingResult]) {
    println!("## 1. Indexing Performance\n");
    println!(
        "| {:<25} | {:>10} | {:>12} | {:>14} | {:>12} | {:>12} | {:>11} |",
        "Scenario",
        "Files",
        "Corpus Size",
        "Avg Index Time",
        "Index Size",
        "N-gram Count",
        "Size Ratio"
    );
    println!(
        "| {:-<25} | {:-<10} | {:-<12} | {:-<14} | {:-<12} | {:-<12} | {:-<11} |",
        "", "", "", "", "", "", ""
    );
    for r in rows {
        println!(
            "| {:<25} | {:>10} | {:>12} | {:>14} | {:>12} | {:>12} | {:>10.1}% |",
            r.scenario,
            r.file_count,
            fmt_bytes(r.corpus_bytes),
            fmt_duration(r.avg_index_time),
            fmt_bytes(r.index_bytes),
            r.ngram_count,
            r.size_ratio_pct,
        );
    }
}

fn print_search_table(rows: &[SearchResult]) {
    println!("## 2. Search Performance\n");
    println!(
        "| {:<25} | {:<18} | {:<7} | {:>16} | {:>12} |",
        "Scenario", "Query", "Kind", "Avg Search Time", "Result Files"
    );
    println!(
        "| {:-<25} | {:-<18} | {:-<7} | {:-<16} | {:-<12} |",
        "", "", "", "", ""
    );
    for r in rows {
        println!(
            "| {:<25} | {:<18} | {:<7} | {:>16} | {:>12} |",
            r.scenario,
            // Truncate long queries for table readability
            if r.query.len() > 18 {
                &r.query[..18]
            } else {
                r.query
            },
            r.kind,
            fmt_duration(r.avg_search_time),
            r.result_count,
        );
    }
}
