# sparse-ngram-search

A fast, low-memory full-text search index for source code, powered by a
**sparse n-gram** algorithm.  It builds a compact inverted index over byte
sequences in your files and answers literal or regex queries in milliseconds —
without loading every file on every search.

[中文文档](README.zh-CN.md)

Use it when you want to:

- jump to symbols or string literals in a large repository;
- give AI agents a fast, machine-readable code search primitive;
- run repeated literal or regex searches without rescanning every file.

---

## How It Works

### Sparse N-gram Algorithm

Traditional n-gram indexing splits text into every possible fixed-width token,
which is expensive in space and time.  This project uses a **weight-guided
boundary** approach instead:

1. Every adjacent byte pair `(a, b)` is assigned a **weight** based on how
   rarely it appears in typical source code (stored in a pre-built frequency
   table).
2. A **boundary** is placed between `i` and `i+1` when both flanking pairs have
   strictly greater weight than every interior pair of the candidate token.
3. This produces tokens that tend to start and end on *unusual* character
   transitions, keeping tokens short and distinctive.

Two extraction modes are provided:

| Mode | Function | Use case |
|---|---|---|
| All n-grams | `build_all` | Build the index (every token in a file) |
| Covering set | `build_covering` | Generate a query (minimal set that covers the pattern) |

### Index Structure

The CLI index is stored in a `.sngi-store/` directory alongside your source files:

```
.sngi-store/
  state.json
  segments/
    segment_0/
      lookup.bin    # sorted array of (ngram_hash → postings_offset)
      postings.bin  # file-id lists per ngram; paths encoded with length prefixes
      meta.json     # version, file count, ngram count, base commit, timestamp
```

Both binary files are memory-mapped (`memmap2`) for O(1) startup.  Lookups use
binary search on the sorted lookup table.

### Query Pipeline

```
pattern
  │
  ├─ literal  →  extract covering n-grams  →  intersect postings  →  verify with memmem
  │
  └─ regex    →  decompose to QueryAtom    →  intersect postings  →  verify with regex engine
                 (AND / OR / Any literals)
```

The postings intersection acts as a **pre-filter**: only candidate files that
contain every required n-gram reach the expensive verification step, giving
near-zero false negatives and dramatically fewer false positives than a plain
grep scan.

### Git-Aware Index

`sparse-ngram-git` adds a two-layer architecture for git repositories:

```
GitAwareIndex
  ├─ base_index   (disk, anchored to a specific commit)
  └─ overlay      (in-memory, tracks add / modify / delete since last sync)
```

On `sync()`, only the files that changed since `base_commit` are re-indexed
into the overlay.  Searches merge results from both layers, with the overlay
masking any deleted or superseded entries from the base.

### Search freshness and the dirty-scan cache

Detecting uncommitted working-tree changes requires scanning the workdir,
which costs seconds on very large repositories.  To keep searches fast, the
scan result is cached in `.sngi-store/dirty_scan.json` and reused while it is
fresh:

- The TTL is adaptive: `min(last scan duration × 10, 30s)`.  Repositories
  with a cheap scan stay effectively always-fresh; large repositories skip
  the rescan for a bounded window.
- Files that become dirty *after* the cached scan may be missed until the
  window expires; files already in the cached list are always re-read and
  re-validated.
- `sngs sync` always forces a full scan (as does `GitAwareIndex::sync()`).
- Set `SNGS_SYNC_TTL_SECS` to pin the TTL (0 disables the cache entirely).

---

## Workspace Layout

```
sparse-ngram-search/
├── crates/
│   ├── core/       # sparse-ngram-core — algorithm, index, and query engine
│   ├── git/        # sparse-ngram-git  — git-aware index (GitAwareIndex)
│   ├── cli/        # sngs              — command-line tool
│   ├── freq-gen/   # freq-gen          — frequency table generator
│   └── bench/      # sparse-ngram-bench — performance benchmark suite
├── Cargo.toml      # workspace manifest
└── README.md
```

---

## Installation

### Build from Source (recommended)

Requires **Rust 1.75+** (edition 2021).

```bash
git clone https://github.com/nano-harness/sparse-ngram-search.git
cd sparse-ngram-search
cargo build -p sngs --release

# Optional: install for the current user
mkdir -p ~/.local/bin
install -m 755 ./target/release/sngs ~/.local/bin/sngs
```

Ensure `~/.local/bin` is on your `PATH`.

---

## Skill Installation

The `sngs-helper` skill is available locally at [`sngs-helper/SKILL.md`](sngs-helper/SKILL.md). Include it in your AI workspace or load it according to your assistant's documentation.

---

## CLI Usage

```
sngs <COMMAND> [OPTIONS]
```

### `init` — Build index

```bash
sngs init                    # index current directory
sngs init --path /my/repo    # index a specific directory
sngs init --no-ignore        # non-Git directories: include ignored files
```

For Git repositories, the index follows Git's tracked and untracked-file model; `--no-ignore` is currently a no-op and produces a warning.

### `search` — Search

```bash
# Literal search
sngs search "fn parse_header"

# Regex search
sngs search -e "impl\s+Display"

# Show only file names (no offsets)
sngs search -l "TODO"

# Limit the number of results
sngs search -n 20 "unsafe"

# Search a specific directory
sngs search --path /my/repo "ConnectionPool"

# Build the index on first search if needed
sngs search --auto-init "ConnectionPool"

# Agent/editor friendly JSON
sngs search --json "ConnectionPool"

# Grep-style lines and path filters
sngs search --grep --glob "src/*" --type rs "ConnectionPool"
```

Default output format:

```
src/parser.rs:3 match(es)
src/proto/header.rs:1 match(es)
[sngs] 2 result(s) in 0.004s
```

### `sync` — Update the index

```bash
sngs sync                    # update the index after files change
sngs sync --path /my/repo
```

Git repositories use incremental overlay updates. Non-Git directories currently perform a full rebuild.

### `status` — Index metadata

```bash
sngs status
```

Example output:

```
Index directory : "/my/repo/.sngi-store"
Version         : 1
Files           : 312
N-gram tokens   : 48291
Last updated    : 93 days after 2024-01-01 (unix epoch 1719878400)
lookup.bin      : 378 KiB
postings.bin    : 512 KiB
```

### `rebuild` — Force full rebuild

```bash
sngs rebuild
sngs rebuild --path /my/repo --no-ignore
```

### Diagnostics and agent-friendly commands

```bash
sngs doctor              # installation and index health checks
sngs warmup              # open the index once to warm OS page cache
sngs explain-index       # explain .sngi-store layout and current metadata
sngs stats               # local opt-out usage statistics
sngs completion zsh      # shell completion script
```

---

## Library Usage

### Core crate

```rust
use sparse_ngram_core::{IndexBuilder, Index, QueryEngine};

// ── Build ──────────────────────────────────────────────────────────────────
let mut builder = IndexBuilder::with_default_weights();
builder.add_file(Path::new("src/main.rs"), b"fn main() { println!(\"hello\"); }")?;
let index = builder.build(Path::new("target/demo-index"))?;

// ── Open (after restart) ───────────────────────────────────────────────────
let index = Index::open(Path::new("target/demo-index"))?;

// ── Search ─────────────────────────────────────────────────────────────────
let engine = QueryEngine::new(&index);

// Literal
let results = engine.search_literal("println", |_id, path| std::fs::read(path).ok())?;

// Regex
let results = engine.search(r"fn\s+\w+\(", |_id, path| std::fs::read(path).ok())?;

for r in &results {
    println!("{}: {:?} offsets", r.path.display(), r.match_offsets);
}
```

### Git-aware crate

```rust
use sparse_ngram_git::GitAwareIndex;

let mut gi = GitAwareIndex::open_or_build(Path::new("/my/repo"))?;

// Sync overlay with working tree changes
let stats = gi.sync()?;
println!("added={} modified={} deleted={}", stats.added, stats.modified, stats.deleted);

// Search across base index + overlay
let results = gi.search("ConnectionPool")?;
```

---

## Benchmarks

A standalone benchmark binary measures indexing time, search time, and index
size across five corpus scenarios.

```bash
cargo run -p sparse-ngram-bench --release
```

The benchmark generates synthetic Rust-like source files, runs each scenario
for several iterations, and prints a Markdown-compatible table to stdout.
Progress is reported to stderr so you can monitor long-running scenarios.

**Scenarios**

| Scenario            | Files | File Size |
|---------------------|------:|----------:|
| 100 files × 1 KiB   |   100 |    1 KiB  |
| 100 files × 4 KiB   |   100 |    4 KiB  |
| 1000 files × 4 KiB  | 1 000 |    4 KiB  |
| 1000 files × 16 KiB | 1 000 |   16 KiB  |
| 5000 files × 4 KiB  | 5 000 |    4 KiB  |

**Sample output** (release build on a modern laptop):

```
## 1. Indexing Performance

| Scenario                  |      Files |  Corpus Size | Avg Index Time |   Index Size | N-gram Count |  Size Ratio |
| ------------------------- | ---------- | ------------ | -------------- | ------------ | ------------ | ----------- |
| 100 files × 1 KiB         |        100 |     117.6 KB |       3.45 ms  |     45.2 KiB |         3821 |       38.4% |
| 100 files × 4 KiB         |        100 |     470.4 KB |      12.10 ms  |    142.7 KiB |        10294 |       30.3% |
| ...                       |        ... |          ... |            ... |          ... |          ... |         ... |

## 2. Search Performance

| Scenario                  | Query              | Kind    |  Avg Search Time | Result Files |
| ------------------------- | ------------------ | ------- | ----------------: | -----------: |
| 100 files × 1 KiB         | fn func_           | literal |          0.3 µs  |          100 |
| 100 files × 1 KiB         | fn \w+_\d+         | regex   |          1.2 µs  |          100 |
| ...                       | ...                | ...     |              ... |          ... |
```

> **Tip:** Run with `--release` — debug builds are 10–20× slower due to
> unoptimised n-gram extraction loops.

---

## Testing

```bash
# All tests (unit + integration)
cargo test --workspace

# Core library only
cargo test -p sparse-ngram-core

# Git-aware index only
cargo test -p sparse-ngram-git

# CLI end-to-end tests
cargo test -p sngs
```

Test coverage includes:

- Literal / regex search hit and miss
- No false negatives guarantee
- Multi-file index consistency
- Rebuild idempotency
- Unicode and emoji content
- `GitAwareIndex` add / modify / delete sync
- All CLI subcommands (`init`, `search`, `sync`, `status`, `rebuild`)

---

## License

MIT
