//! `sngs` – Sparse N-gram Search CLI.
//!
//! # Commands
//!
//! ```text
//! sngs init [--path <repo>]              # Initialise index for a git repo
//! sngs sync [--path <repo>]              # Sync index to current HEAD + working tree
//! sngs search <pattern>                  # Literal search
//! sngs search -e <regex>                 # Regex search
//! sngs status [--path <repo>]            # Show index metadata
//! sngs rebuild [--path <repo>]           # Force full rebuild
//! sngs doctor [--path <repo>]            # Diagnose install and index state
//! sngs stats                             # Show local usage statistics
//! sngs update                            # Self-update to the latest release
//! sngs completion <bash|zsh|fish|...>    # Generate shell completion scripts
//! ```

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{generate, Shell};
use ignore::WalkBuilder;
use rayon::prelude::*;
use sparse_ngram_core::index::IndexBuilder;
use sparse_ngram_core::segment::{SegmentLevel, SegmentManager};
use sparse_ngram_core::storage::should_skip_dir;
use sparse_ngram_core::storage::{set_external_index_base, IndexStorePaths, INDEX_STORE_DIR};
use sparse_ngram_core::{default_weight_fn, file_ngram_hashes, FrequencyWeight};
use sparse_ngram_git::GitAwareIndex;

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "sngs",
    about = "Sparse N-gram Search — fast regex/literal index for source code",
    version,
    propagate_version = true
)]
struct Cli {
    /// Timeout in seconds (0 = no timeout).
    #[arg(long, global = true, default_value = "0", value_name = "SECS")]
    timeout: u64,

    /// External index directory (default: inside repo).
    #[arg(long, global = true, value_name = "DIR")]
    index_dir: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Initialise a search index for the given directory (or current dir).
    Init {
        /// Path to the directory to index (default: current directory).
        #[arg(short, long, default_value = ".")]
        path: PathBuf,
        /// Include ignored files (non-Git directories only).
        #[arg(long)]
        no_ignore: bool,
    },

    /// Sync the index (incremental for Git repositories, full rebuild otherwise).
    Sync {
        /// Path to the indexed directory.
        #[arg(short, long, default_value = ".")]
        path: PathBuf,
    },

    /// Search for a pattern in the indexed directory.
    Search {
        /// The pattern to search for.
        pattern: String,
        /// Treat <pattern> as a regular expression.
        #[arg(short = 'e', long)]
        regex: bool,
        /// Path to the indexed directory.
        #[arg(short, long, default_value = ".")]
        path: PathBuf,
        /// Print only file names (no offsets).
        #[arg(short = 'l', long, alias = "files-with-matches")]
        files_only: bool,
        /// Maximum number of results to show (0 = unlimited).
        #[arg(short = 'n', long, default_value = "0")]
        max_results: usize,
        /// Show line numbers in the output.
        #[arg(long)]
        line_numbers: bool,
        /// Show N lines of context around each match (like grep -C).
        #[arg(short = 'C', long)]
        context: Option<usize>,
        /// Build an index automatically if none exists.
        #[arg(long)]
        auto_init: bool,
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
        /// Emit grep-style file:line:content output.
        #[arg(long, alias = "rg-style")]
        grep: bool,
        /// Print match counts per file.
        #[arg(short = 'c', long)]
        count: bool,
        /// Include only paths matching this glob-like pattern (* and ? supported).
        #[arg(long = "glob", value_name = "PATTERN")]
        globs: Vec<String>,
        /// Exclude paths matching this glob-like pattern (* and ? supported).
        #[arg(long = "exclude", value_name = "PATTERN")]
        excludes: Vec<String>,
        /// Include only files with this extension/type (for example: rs, py, js).
        #[arg(long = "type", value_name = "EXT")]
        type_filters: Vec<String>,
    },

    /// Show index status and metadata.
    Status {
        /// Path to the indexed directory.
        #[arg(short, long, default_value = ".")]
        path: PathBuf,
    },

    /// Force a full index rebuild.
    Rebuild {
        /// Path to the indexed directory.
        #[arg(short, long, default_value = ".")]
        path: PathBuf,
        /// Include ignored files (non-Git directories only).
        #[arg(long)]
        no_ignore: bool,
    },

    /// Diagnose installation, index state, and common setup issues.
    Doctor {
        /// Path to inspect.
        #[arg(short, long, default_value = ".")]
        path: PathBuf,
    },

    /// Show local, opt-out usage statistics.
    Stats,

    /// Open the index once so OS page cache is warmed for subsequent searches.
    Warmup {
        /// Path to the indexed directory.
        #[arg(short, long, default_value = ".")]
        path: PathBuf,
    },

    /// Explain index layout and current repository index metadata.
    ExplainIndex {
        /// Path to the indexed directory.
        #[arg(short, long, default_value = ".")]
        path: PathBuf,
    },

    /// Generate shell completion scripts.
    Completion {
        /// The shell to generate completions for.
        #[arg(value_enum)]
        shell: Shell,
    },
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    // T15: Exit codes match ripgrep convention
    // 0 = matched, 1 = no match, 2 = error
    std::process::exit(run());
}

fn run() -> i32 {
    let cli = Cli::parse();

    // Set external index directory if specified via CLI
    set_external_index_base(cli.index_dir.clone());

    let timeout_secs = cli.timeout;
    let command = cli.command;

    if timeout_secs == 0 {
        return match dispatch(command, timeout_secs) {
            Ok(had_matches) => {
                if had_matches {
                    0
                } else {
                    1
                }
            }
            Err(e) => {
                eprintln!("error: {e:#}");
                2
            }
        };
    }

    // Run the command in a worker thread; wait up to `timeout_secs` seconds.
    let (tx, rx) = std::sync::mpsc::channel::<Result<bool>>();
    let timeout_secs = cli.timeout;
    std::thread::spawn(move || {
        let _ = tx.send(dispatch(command, timeout_secs));
    });

    match rx.recv_timeout(Duration::from_secs(timeout_secs)) {
        Ok(Ok(had_matches)) => {
            if had_matches {
                0
            } else {
                1
            }
        }
        Ok(Err(e)) => {
            eprintln!("error: {e:#}");
            2
        }
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            eprintln!("error: command timed out after {}s", timeout_secs);
            2
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            eprintln!("error: worker thread panicked");
            2
        }
    }
}

fn dispatch(command: Command, timeout_secs: u64) -> Result<bool> {
    // F5: Create cancellation token with timeout if specified
    let cancel_token = if timeout_secs > 0 {
        Some(sparse_ngram_core::CancellationToken::with_timeout(
            Duration::from_secs(timeout_secs),
        ))
    } else {
        None
    };

    match command {
        Command::Init { path, no_ignore } => cmd_init(&path, no_ignore).map(|_| true),
        Command::Sync { path } => cmd_sync(&path).map(|_| true),
        Command::Search {
            pattern,
            regex,
            path,
            files_only,
            max_results,
            line_numbers,
            context,
            auto_init,
            json,
            grep,
            count,
            globs,
            excludes,
            type_filters,
        } => cmd_search(
            &path,
            &pattern,
            regex,
            files_only,
            max_results,
            line_numbers,
            context,
            auto_init,
            json,
            grep,
            count,
            &globs,
            &excludes,
            &type_filters,
            cancel_token.as_ref(),
        ),
        Command::Status { path } => cmd_status(&path).map(|_| true),
        Command::Rebuild { path, no_ignore } => cmd_rebuild(&path, no_ignore).map(|_| true),
        Command::Doctor { path } => cmd_doctor(&path).map(|_| true),
        Command::Stats => cmd_stats().map(|_| true),
        Command::Warmup { path } => cmd_warmup(&path).map(|_| true),
        Command::ExplainIndex { path } => cmd_explain_index(&path).map(|_| true),
        Command::Completion { shell } => cmd_completion(shell).map(|_| true),
    }
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

fn cmd_init(root: &Path, no_ignore: bool) -> Result<()> {
    let root = root.canonicalize().context("canonicalize path")?;
    let store = IndexStorePaths::new(&root);

    if is_git_repo(&root) {
        // F10: Warn if --no-ignore is used in a git repository
        if no_ignore {
            eprintln!("warning: --no-ignore is currently a no-op for git-tracked repositories");
        }

        // Only the default workspace-local store needs an ignore entry.
        // External indexes must also work with an unmodified, read-only repo.
        if store.root == root.join(INDEX_STORE_DIR) {
            add_to_gitignore(&root)?;
        }

        let start = Instant::now();
        let gi = GitAwareIndex::open_or_build(&root)?;
        println!(
            "Git-aware index ready at {:?} (base {}, {} segment(s)) in {:.2}s",
            store.root,
            short_commit(gi.base_commit()),
            gi.segment_count(),
            start.elapsed().as_secs_f64()
        );
        return Ok(());
    }

    // Check if SegmentManager state exists
    if store.root.join("state.json").exists() {
        eprintln!(
            "index already exists at {:?}; use `sngs rebuild` to force rebuild",
            store.root
        );
        return Ok(());
    }

    println!("Building index for {:?} …", root);
    let start = Instant::now();

    // Build the index using SegmentManager
    let mut builder = IndexBuilder::with_default_weights();
    collect_files_into_builder(&root, no_ignore, &mut builder)?;

    // Create SegmentManager and add the initial segment
    let mut mgr = SegmentManager::create(&store.root)?;
    mgr.add_segment(builder, None, SegmentLevel::L3)?;

    println!(
        "Done: {} files, {} n-gram tokens in {:.2}s",
        mgr.segments()[0].index.file_count(),
        mgr.segments()[0].index.ngram_count(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn cmd_rebuild(root: &Path, no_ignore: bool) -> Result<()> {
    let root = root.canonicalize().context("canonicalize path")?;
    let store = IndexStorePaths::new(&root);

    // F10: Warn if --no-ignore is used in a git repository
    if no_ignore && is_git_repo(&root) {
        eprintln!("warning: --no-ignore is currently a no-op for git-tracked repositories");
    }

    // Git repositories must rebuild from the HEAD tree (same path as
    // `sngs init`), not from a filesystem walk: the walk applies .gitignore
    // rules and silently drops tracked-but-ignored files (e.g. gerrit's
    // tools/node_tools/**), and it would destroy the git-aware metadata
    // (head.txt) that incremental sync relies on.
    if is_git_repo(&root) {
        let start = Instant::now();
        if store.root.exists() {
            std::fs::remove_dir_all(&store.root)
                .with_context(|| format!("remove old store at {:?}", store.root))?;
        }
        let gi = GitAwareIndex::open_or_build(&root)?;
        println!(
            "Git-aware index rebuilt at {:?} (base {}, {} segment(s)) in {:.2}s",
            store.root,
            short_commit(gi.base_commit()),
            gi.segment_count(),
            start.elapsed().as_secs_f64()
        );
        return Ok(());
    }

    println!("Rebuilding index for {:?} …", root);
    let start = Instant::now();

    // Build the index using SegmentManager
    let mut builder = IndexBuilder::with_default_weights();
    collect_files_into_builder(&root, no_ignore, &mut builder)?;

    // T17: Build to {store.root}.new/ then swap
    let new_store = store.root.with_extension("new");
    if new_store.exists() {
        std::fs::remove_dir_all(&new_store)
            .with_context(|| format!("remove existing temp store at {:?}", new_store))?;
    }

    let mut mgr = SegmentManager::create(&new_store)?;
    mgr.add_segment(builder, None, SegmentLevel::L3)?;

    // Swap: remove old, rename new
    if store.root.exists() {
        std::fs::remove_dir_all(&store.root)
            .with_context(|| format!("remove old store at {:?}", store.root))?;
    }
    std::fs::rename(&new_store, &store.root)
        .with_context(|| format!("rename {:?} to {:?}", new_store, store.root))?;

    println!(
        "Done: {} files, {} n-gram tokens in {:.2}s",
        mgr.segments()[0].index.file_count(),
        mgr.segments()[0].index.ngram_count(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn cmd_completion(shell: Shell) -> Result<()> {
    let mut cmd = Cli::command();
    generate(shell, &mut cmd, "sngs", &mut std::io::stdout());
    Ok(())
}

/// Returns `Ok(true)` when `latest` is strictly newer than `current`.
/// Both strings should be bare semver like "1.2.3" (no leading "v").
/// Returns an error if either string cannot be parsed as 3-component semver.
#[allow(dead_code)]
fn semver_is_newer(latest: &str, current: &str) -> Result<bool> {
    fn parse(s: &str) -> Option<[u64; 3]> {
        let mut parts = s.splitn(3, '.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next()?.parse().ok()?;
        // All three components must be present; strip any pre-release suffix.
        let patch_str = parts.next()?;
        let patch = patch_str.split('-').next()?.parse().ok()?;
        Some([major, minor, patch])
    }
    let l = parse(latest).with_context(|| format!("cannot parse latest version: {latest:?}"))?;
    let c = parse(current).with_context(|| format!("cannot parse current version: {current:?}"))?;
    Ok(l > c)
}

fn cmd_sync(root: &Path) -> Result<()> {
    let root = root.canonicalize().context("canonicalize path")?;
    let store = IndexStorePaths::new(&root);

    if is_git_repo(&root) {
        let start = Instant::now();
        let mut gi = GitAwareIndex::open_or_build(&root)?;
        let stats = gi.sync()?;
        println!(
            "Done: added={}, modified={}, deleted={}, compacted={}, base={}, segments={} in {:.2}s",
            stats.added,
            stats.modified,
            stats.deleted,
            stats.compacted,
            short_commit(gi.base_commit()),
            gi.segment_count(),
            start.elapsed().as_secs_f64()
        );
        return Ok(());
    }

    // Check if store exists
    if !store.root.join("state.json").exists() {
        bail!("no index found at {:?}; run `sngs init` first", store.root);
    }

    println!("Syncing index …");
    let start = Instant::now();

    // Open the store, rebuild all segments
    let mgr = SegmentManager::open(&store.root)?;

    // For now, do a full rebuild (same as before)
    // TODO: Implement incremental sync for non-git directories
    let mut builder = IndexBuilder::with_default_weights();
    collect_files_into_builder(&root, false, &mut builder)?;

    // Remove old store and create fresh one
    drop(mgr); // Close before removing
    std::fs::remove_dir_all(&store.root)
        .with_context(|| format!("remove store at {:?}", store.root))?;

    let mut new_mgr = SegmentManager::create(&store.root)?;
    new_mgr.add_segment(builder, None, SegmentLevel::L3)?;

    println!(
        "Done: {} files, {} n-gram tokens in {:.2}s",
        new_mgr.segments()[0].index.file_count(),
        new_mgr.segments()[0].index.ngram_count(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn cmd_status(root: &Path) -> Result<()> {
    let root = root.canonicalize().context("canonicalize path")?;
    let store = IndexStorePaths::new(&root);

    if is_git_repo(&root) && store.root.join("head.txt").exists() {
        // F7: Use read-only mode for status command
        let gi = GitAwareIndex::open_readonly(&root)?;
        println!("Index directory : {:?}", store.root);
        println!("Mode            : git-aware");
        println!("Base commit     : {}", gi.base_commit());
        println!("Segments        : {}", gi.segment_count());
        println!("Overlay modified: {}", gi.overlay_modified_count());
        println!("Overlay deleted : {}", gi.overlay_deleted_count());
        return Ok(());
    }

    // Check if store exists
    if !store.root.join("state.json").exists() {
        bail!("no index found at {:?}; run `sngs init` first", store.root);
    }

    // Open SegmentManager and aggregate stats
    let mgr = SegmentManager::open(&store.root)?;

    let mut total_files = 0;
    let mut total_ngrams = 0;

    for seg in mgr.segments() {
        total_files += seg.index.file_count();
        total_ngrams += seg.index.ngram_count();
    }

    println!("Index directory : {:?}", store.root);
    println!("Segments        : {}", mgr.segment_count());
    println!("Files           : {}", total_files);
    println!("N-gram tokens   : {}", total_ngrams);

    // Show per-segment details
    for seg in mgr.segments() {
        let level_str = match seg.level {
            SegmentLevel::L1 => "L1 (WAL)",
            SegmentLevel::L2 => "L2 (Compacted)",
            SegmentLevel::L3 => "L3 (Base)",
        };
        println!(
            "  Segment {} ({}): {} files, {} n-grams",
            seg.id,
            level_str,
            seg.index.file_count(),
            seg.index.ngram_count()
        );
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_search(
    root: &Path,
    pattern: &str,
    is_regex: bool,
    files_only: bool,
    max_results: usize,
    line_numbers: bool,
    context: Option<usize>,
    auto_init: bool,
    json: bool,
    grep: bool,
    count: bool,
    globs: &[String],
    excludes: &[String],
    type_filters: &[String],
    cancel: Option<&sparse_ngram_core::CancellationToken>,
) -> Result<bool> {
    let root = root.canonicalize().context("canonicalize path")?;
    let store = IndexStorePaths::new(&root);

    if !store.root.join("state.json").exists() {
        if auto_init || is_git_repo(&root) {
            eprintln!("no index found at {:?}; building one now …", store.root);
            cmd_init(&root, false)?;
        } else {
            bail!(
                "no index found at {:?}; run `sngs init` first or retry with `sngs search --auto-init ...`",
                store.root
            );
        }
    }

    if is_git_repo(&root) && store.root.join("head.txt").exists() {
        return cmd_search_git(
            &root,
            pattern,
            is_regex,
            files_only,
            max_results,
            line_numbers,
            context,
            json,
            grep,
            count,
            globs,
            excludes,
            type_filters,
            cancel,
        );
    }

    // Open SegmentManager and search across all segments
    let mgr = SegmentManager::open(&store.root)?;

    // `context` is Some(_) when -C was explicitly passed; `line_numbers` also activates
    // per-line output.
    let ctx_lines = context.unwrap_or(0);
    let need_content = (line_numbers || context.is_some() || grep || json) && !files_only;

    let start = Instant::now();
    let root2 = root.clone();

    // Cache file contents while verifying matches so we can reuse the same bytes for
    // line-number / context output without a second round of disk reads.
    // Mutex because verification may run in parallel.
    let content_cache: std::sync::Mutex<std::collections::HashMap<PathBuf, Vec<u8>>> =
        std::sync::Mutex::new(std::collections::HashMap::new());

    // FP-3: Route through SegmentManager::search so the non-git path gets the
    // same deletion-bitmap filtering and newest-first dedup semantics as the
    // git path, instead of merging per-segment QueryEngine results by hand.
    // SegmentManager::search only accepts a regex, so literal patterns are
    // escaped first (same approach as cmd_search_git).
    let query = if is_regex {
        pattern.to_string()
    } else {
        regex::escape(pattern)
    };
    let load_content = |path: &Path| -> Option<Vec<u8>> {
        let full = root2.join(path);
        match std::fs::read(&full) {
            Ok(bytes) => {
                if need_content {
                    content_cache
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .insert(path.to_path_buf(), bytes.clone());
                }
                Some(bytes)
            }
            Err(_) => None,
        }
    };
    // files_only with no other display flag needs no match offsets — stop
    // verification at the first match per file.  (-l combined with -c/--json/
    // -C/--grep/--line-numbers still consumes offsets, so keep the full path.)
    let files_only_fast =
        files_only && !count && !json && !line_numbers && context.is_none() && !grep;
    let mut results = if files_only_fast {
        mgr.search_files_only(&query, load_content, cancel)?
    } else {
        mgr.search(&query, load_content, cancel)?
    };
    apply_path_filters(&mut results, globs, excludes, type_filters);
    results.sort_by(|a, b| a.path.cmp(&b.path));

    let content_cache = content_cache
        .into_inner()
        .unwrap_or_else(|e| e.into_inner());

    let elapsed = start.elapsed();
    let total = results.len();
    let shown = if max_results == 0 {
        total
    } else {
        total.min(max_results)
    };

    print_results(
        &results,
        shown,
        files_only,
        need_content,
        ctx_lines,
        count,
        json,
        grep,
        &content_cache,
    );

    if max_results > 0 && total > max_results {
        println!(
            "… {} more results (use -n 0 to show all)",
            total - max_results
        );
    }

    eprintln!(
        "[sngs] {} result(s) in {:.3}s",
        total,
        elapsed.as_secs_f64()
    );
    record_search_stat(total, elapsed);

    // T15: Return true if matches were found, false otherwise
    Ok(total > 0)
}

#[allow(clippy::too_many_arguments)]
fn cmd_search_git(
    root: &Path,
    pattern: &str,
    is_regex: bool,
    files_only: bool,
    max_results: usize,
    line_numbers: bool,
    context: Option<usize>,
    json: bool,
    grep: bool,
    count: bool,
    globs: &[String],
    excludes: &[String],
    type_filters: &[String],
    cancel: Option<&sparse_ngram_core::CancellationToken>,
) -> Result<bool> {
    let start = Instant::now();
    let gi = GitAwareIndex::open_or_build(root)?;
    let query = if is_regex {
        pattern.to_string()
    } else {
        regex::escape(pattern)
    };
    // files_only with no other display flag needs no match offsets — stop
    // verification at the first match per file.  (-l combined with -c/--json/
    // -C/--grep/--line-numbers still consumes offsets, so keep the full path.)
    let files_only_fast =
        files_only && !count && !json && !line_numbers && context.is_none() && !grep;
    let mut results = if files_only_fast {
        gi.search_files_only(&query, cancel)?
    } else {
        gi.search(&query, cancel)?
    };
    apply_path_filters(&mut results, globs, excludes, type_filters);
    results.sort_by(|a, b| a.path.cmp(&b.path));

    let ctx_lines = context.unwrap_or(0);
    let need_content = (line_numbers || context.is_some() || grep || json) && !files_only;
    let mut content_cache = std::collections::HashMap::new();
    if need_content {
        for result in &results {
            if let Ok(bytes) = std::fs::read(root.join(&result.path)) {
                content_cache.insert(result.path.clone(), bytes);
            }
        }
    }

    let total = results.len();
    let shown = if max_results == 0 {
        total
    } else {
        total.min(max_results)
    };
    print_results(
        &results,
        shown,
        files_only,
        need_content,
        ctx_lines,
        count,
        json,
        grep,
        &content_cache,
    );
    if max_results > 0 && total > max_results {
        println!(
            "… {} more results (use -n 0 to show all)",
            total - max_results
        );
    }
    let elapsed = start.elapsed();
    eprintln!(
        "[sngs] {} result(s) in {:.3}s",
        total,
        elapsed.as_secs_f64()
    );
    record_search_stat(total, elapsed);
    // T15: Return true if matches were found, false otherwise
    Ok(total > 0)
}

#[allow(clippy::too_many_arguments)]
fn print_results(
    results: &[sparse_ngram_core::query::SearchResult],
    shown: usize,
    files_only: bool,
    need_content: bool,
    ctx_lines: usize,
    count: bool,
    json: bool,
    grep: bool,
    content_cache: &std::collections::HashMap<PathBuf, Vec<u8>>,
) {
    if json {
        let rows: Vec<_> = results
            .iter()
            .take(shown)
            .map(|result| {
                let lines = content_cache
                    .get(&result.path)
                    .map(|bytes| json_line_matches(bytes, &result.match_offsets))
                    .unwrap_or_default();
                serde_json::json!({
                    "path": result.path,
                    "match_count": result.match_offsets.len(),
                    "offsets": result.match_offsets,
                    "lines": lines,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string(&rows).unwrap_or_else(|_| "[]".into())
        );
        return;
    }

    // T14: Implement --grep flag for ripgrep-style output
    if grep {
        for result in results.iter().take(shown) {
            if let Some(bytes) = content_cache.get(&result.path) {
                print_grep_output(
                    bytes,
                    &result.match_offsets,
                    &result.path.display().to_string(),
                );
            }
        }
        return;
    }

    for result in results.iter().take(shown) {
        if files_only {
            println!("{}", result.path.display());
        } else if count {
            println!("{}:{}", result.path.display(), result.match_offsets.len());
        } else if need_content {
            if let Some(bytes) = content_cache.get(&result.path) {
                print_line_output(
                    bytes,
                    &result.match_offsets,
                    ctx_lines,
                    &result.path.display().to_string(),
                );
            } else if result.match_offsets.is_empty() {
                println!("{}:unverified", result.path.display());
            } else {
                println!(
                    "{}:{} match(es)",
                    result.path.display(),
                    result.match_offsets.len()
                );
            }
        } else {
            println!(
                "{}:{} match(es)",
                result.path.display(),
                result.match_offsets.len()
            );
        }
    }
}

fn cmd_doctor(root: &Path) -> Result<()> {
    let root = root.canonicalize().context("canonicalize path")?;
    let store = IndexStorePaths::new(&root);
    println!("sngs version    : {}", env!("CARGO_PKG_VERSION"));
    println!("Executable      : {:?}", std::env::current_exe().ok());
    println!("Workspace       : {:?}", root);
    println!(
        "Repository type : {}",
        if is_git_repo(&root) {
            "git"
        } else {
            "plain directory"
        }
    );
    println!("Index directory : {:?}", store.root);
    println!(
        "Index exists    : {}",
        store.root.join("state.json").exists()
    );
    println!(
        "Index size      : {} bytes",
        dir_size(&store.root).unwrap_or(0)
    );
    if is_git_repo(&root) && store.root.join("head.txt").exists() {
        let gi = GitAwareIndex::open_or_build(&root)?;
        println!("Mode            : git-aware");
        println!("Base commit     : {}", gi.base_commit());
        println!("Segments        : {}", gi.segment_count());
        println!("Overlay modified: {}", gi.overlay_modified_count());
        println!("Overlay deleted : {}", gi.overlay_deleted_count());
    } else if store.root.join("state.json").exists() {
        let mgr = SegmentManager::open(&store.root)?;
        println!("Mode            : directory");
        println!("Segments        : {}", mgr.segment_count());
    } else {
        println!("Next step       : run `sngs init` or `sngs search --auto-init <pattern>`");
    }
    println!(
        "Stats           : {}",
        if stats_disabled() {
            "disabled by SNGS_NO_STATS"
        } else {
            "enabled locally"
        }
    );
    Ok(())
}

fn cmd_stats() -> Result<()> {
    let stats = read_stats();
    let searches = stats
        .get("searches")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let total_ms = stats
        .get("total_ms")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    let last_results = stats
        .get("last_result_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let last_search_at = stats
        .get("last_search_at")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("never");
    let avg_ms = if searches == 0 {
        0.0
    } else {
        total_ms / searches as f64
    };

    println!("Searches        : {searches}");
    println!("Average latency : {avg_ms:.3} ms");
    println!("Last results    : {last_results}");
    println!("Last search at  : {last_search_at}");
    println!("Opt out         : set SNGS_NO_STATS=1");
    Ok(())
}

fn cmd_warmup(root: &Path) -> Result<()> {
    let root = root.canonicalize().context("canonicalize path")?;
    let store = IndexStorePaths::new(&root);
    if is_git_repo(&root) && store.root.join("head.txt").exists() {
        // F7: Use read-only mode for warmup command
        let gi = GitAwareIndex::open_readonly(&root)?;
        println!(
            "Warmed git-aware index ({} segment(s)).",
            gi.segment_count()
        );
    } else {
        let mgr = SegmentManager::open(&store.root)
            .with_context(|| format!("open index at {:?}", store.root))?;
        println!(
            "Warmed directory index ({} segment(s)).",
            mgr.segment_count()
        );
    }
    Ok(())
}

fn cmd_explain_index(root: &Path) -> Result<()> {
    println!("sngs stores its index in .sngi-store/ next to the searched workspace.");
    println!("segments/ contains immutable sparse n-gram segments; state.json tracks them.");
    println!("In git repositories, head.txt records the indexed commit and dirty files are handled by an overlay.");
    cmd_status(root)
}

fn json_line_matches(content: &[u8], offsets: &[usize]) -> Vec<serde_json::Value> {
    let line_starts = build_line_starts(content);
    offsets
        .iter()
        .map(|&offset| {
            let offset = offset.min(content.len());
            let line = line_num_for_offset(&line_starts, offset);
            serde_json::json!({
                "line": line,
                "text": line_at(content, &line_starts, line),
            })
        })
        .collect()
}

fn apply_path_filters(
    results: &mut Vec<sparse_ngram_core::query::SearchResult>,
    globs: &[String],
    excludes: &[String],
    type_filters: &[String],
) {
    if globs.is_empty() && excludes.is_empty() && type_filters.is_empty() {
        return;
    }
    results.retain(|r| {
        let path = r.path.to_string_lossy();
        let included = globs.is_empty() || globs.iter().any(|g| wildcard_match(g, &path));
        let excluded = excludes.iter().any(|g| wildcard_match(g, &path));
        let type_ok = type_filters.is_empty()
            || r.path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|ext| {
                    type_filters
                        .iter()
                        .any(|t| ext.eq_ignore_ascii_case(t.trim_start_matches('.')))
                });
        included && !excluded && type_ok
    });
}

fn wildcard_match(pattern: &str, text: &str) -> bool {
    let p = pattern.as_bytes();
    let t = text.as_bytes();
    let (mut pi, mut ti) = (0usize, 0usize);
    let mut star: Option<usize> = None;
    let mut match_i = 0usize;

    while ti < t.len() {
        if pi < p.len() && (p[pi] == b'?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == b'*' {
            star = Some(pi);
            match_i = ti;
            pi += 1;
        } else if let Some(star_i) = star {
            pi = star_i + 1;
            match_i += 1;
            ti = match_i;
        } else {
            return false;
        }
    }

    while pi < p.len() && p[pi] == b'*' {
        pi += 1;
    }
    pi == p.len()
}

fn is_git_repo(root: &Path) -> bool {
    root.join(".git").exists()
}

/// T16: Add .sngi-store/ to .gitignore if not already present
fn add_to_gitignore(root: &Path) -> Result<()> {
    let gitignore_path = root.join(".gitignore");
    let entry = ".sngi-store/";

    // Read existing .gitignore or create empty
    let content = if gitignore_path.exists() {
        std::fs::read_to_string(&gitignore_path)
            .with_context(|| format!("read {:?}", gitignore_path))?
    } else {
        String::new()
    };

    // Check if entry already exists
    if content.lines().any(|line| line.trim() == entry) {
        return Ok(());
    }

    // Append entry
    let new_content = if content.is_empty() || content.ends_with('\n') {
        format!("{}{}\n", content, entry)
    } else {
        format!("{}\n{}\n", content, entry)
    };

    std::fs::write(&gitignore_path, new_content)
        .with_context(|| format!("write {:?}", gitignore_path))?;

    Ok(())
}

fn short_commit(commit: &str) -> &str {
    commit
        .char_indices()
        .nth(12)
        .map(|(idx, _)| &commit[..idx])
        .unwrap_or(commit)
}

fn dir_size(path: &Path) -> Result<u64> {
    if !path.exists() {
        return Ok(0);
    }
    let mut total = 0;
    for entry in std::fs::read_dir(path).with_context(|| format!("read dir {:?}", path))? {
        let entry = entry?;
        let meta = entry.metadata()?;
        if meta.is_dir() {
            total += dir_size(&entry.path())?;
        } else {
            total += meta.len();
        }
    }
    Ok(total)
}

fn stats_path() -> Option<PathBuf> {
    dirs::data_local_dir().map(|d| d.join("sngs").join("stats.json"))
}

fn stats_disabled() -> bool {
    std::env::var_os("SNGS_NO_STATS").is_some()
}

fn read_stats() -> serde_json::Value {
    let Some(path) = stats_path() else {
        return serde_json::json!({});
    };
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| serde_json::json!({}))
}

fn record_search_stat(result_count: usize, elapsed: Duration) {
    if stats_disabled() {
        return;
    }
    let Some(path) = stats_path() else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let stats = read_stats();
    let searches = stats
        .get("searches")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0)
        + 1;
    let total_ms = stats
        .get("total_ms")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0)
        + elapsed.as_secs_f64() * 1000.0;
    let payload = serde_json::json!({
        "searches": searches,
        "total_ms": total_ms,
        "last_result_count": result_count,
        "last_search_at": unix_timestamp(),
    });
    let _ = std::fs::write(
        path,
        serde_json::to_vec_pretty(&payload).unwrap_or_default(),
    );
}

fn unix_timestamp() -> u64 {
    // If the system clock is somehow before the Unix epoch, keep statistics
    // writable by recording a conservative timestamp instead of failing search.
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Line-output helpers
// ---------------------------------------------------------------------------

/// Print per-line match output with optional context, using a precomputed line index
/// for O(log n) lookup per match instead of O(file_size) per match.
fn print_line_output(
    content: &[u8],
    match_offsets: &[usize],
    ctx_lines: usize,
    file_display: &str,
) {
    // Build line index once: `line_starts[i]` = byte offset where line `i+1` begins.
    let line_starts = build_line_starts(content);
    let total_lines = line_starts.len();

    // Track the last line number that has already been output so we can:
    //   (a) avoid reprinting overlapping context, and
    //   (b) emit a "--" separator when there is a genuine gap between groups.
    let mut last_printed: Option<usize> = None;

    for &offset in match_offsets {
        let offset = offset.min(content.len());
        let ln = line_num_for_offset(&line_starts, offset);

        // Context window [first, last] (1-based, clamped to file bounds).
        let first = ln.saturating_sub(ctx_lines).max(1);
        let last = (ln + ctx_lines).min(total_lines);

        // Determine the starting line for this iteration and whether to print a separator.
        let print_from = match last_printed {
            None => first,
            Some(prev) if first > prev + 1 => {
                // There is a gap between the previous group and this one.
                println!("--");
                first
            }
            Some(prev) => {
                // Overlapping or adjacent: skip already-printed lines.
                (prev + 1).max(first)
            }
        };

        for ln in print_from..=last {
            println!(
                "{}:{}:{}",
                file_display,
                ln,
                line_at(content, &line_starts, ln)
            );
        }

        last_printed = Some(last);
    }
}

/// Return a `Vec` of byte offsets where each line begins (entry 0 is always 0).
/// A trailing newline does **not** produce a phantom empty line entry.
fn build_line_starts(content: &[u8]) -> Vec<usize> {
    let mut starts = vec![0usize];
    for (i, &b) in content.iter().enumerate() {
        if b == b'\n' && i + 1 < content.len() {
            starts.push(i + 1);
        }
    }
    starts
}

/// Return the 1-based line number for `offset`, given the precomputed `line_starts`.
fn line_num_for_offset(line_starts: &[usize], offset: usize) -> usize {
    match line_starts.binary_search(&offset) {
        // `offset` is exactly the start of a line.
        Ok(i) => i + 1,
        // `i` = number of line-starts ≤ offset, which equals the 1-based line number.
        Err(i) => i,
    }
}

/// Return the content of 1-based line `line_num` as a `String` without the trailing newline.
fn line_at(content: &[u8], line_starts: &[usize], line_num: usize) -> String {
    let idx = line_num.saturating_sub(1); // 0-based
    if idx >= line_starts.len() {
        return String::new();
    }
    let start = line_starts[idx];
    // End = start of next line minus 1 (the '\n' byte), or end of content.
    let end = line_starts
        .get(idx + 1)
        .copied()
        .map(|s| s - 1)
        .unwrap_or(content.len());
    // Also strip '\r' for CRLF files.
    let bytes = &content[start..end];
    let bytes = bytes.strip_suffix(b"\r").unwrap_or(bytes);
    String::from_utf8_lossy(bytes).into_owned()
}

/// T14: Print grep-style output (path:line:column:content)
fn print_grep_output(content: &[u8], match_offsets: &[usize], file_display: &str) {
    let line_starts = build_line_starts(content);

    for &offset in match_offsets {
        let offset = offset.min(content.len());
        let ln = line_num_for_offset(&line_starts, offset);
        let line_content = line_at(content, &line_starts, ln);

        // Calculate column (1-based byte offset from line start)
        let line_start = if ln > 0 { line_starts[ln - 1] } else { 0 };
        let column = offset.saturating_sub(line_start) + 1;

        println!("{}:{}:{}:{}", file_display, ln, column, line_content);
    }
}

// ---------------------------------------------------------------------------
// Index construction helpers
// ---------------------------------------------------------------------------

/// Collect all text files from `root` into `builder`.
///
/// Pipeline:
/// 1. Walk the directory tree (cheap) and gather candidate file paths.
/// 2. Read file contents and extract n-gram hash sets **in parallel** (Rayon).
/// 3. Merge the per-file hash sets via [`IndexBuilder::from_file_hashes`]
///    (parallel chunk merge).
fn collect_files_into_builder(
    root: &Path,
    no_ignore: bool,
    builder: &mut IndexBuilder,
) -> Result<()> {
    let walk = WalkBuilder::new(root)
        .hidden(false)
        .ignore(!no_ignore)
        .git_ignore(!no_ignore)
        .git_global(!no_ignore)
        .git_exclude(!no_ignore)
        // Skip directories that should never be indexed.
        .filter_entry(|entry| {
            if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                return entry
                    .file_name()
                    .to_str()
                    .map(|name| !should_skip_dir(name))
                    .unwrap_or(false);
            }
            true
        })
        .build();

    // ── Phase 1: collect candidate paths ────────────────────────────────────
    let mut rel_paths: Vec<PathBuf> = Vec::new();
    let mut skipped = 0usize;

    for entry in walk {
        let entry = entry.context("walk entry")?;
        if entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
            let path = entry.path();
            // Skip likely-binary files.
            if is_likely_binary(path) {
                skipped += 1;
                continue;
            }
            // Store path relative to root.
            let rel = path.strip_prefix(root).unwrap_or(path);
            rel_paths.push(rel.to_path_buf());
        }
    }

    // ── Phase 2: read + extract n-gram hashes in parallel ───────────────────
    // Concrete FrequencyWeight so weight lookups are inlined (monomorphised).
    let wf = FrequencyWeight::with_utf8_smoothing();
    let per_file: Vec<Option<Vec<u64>>> = rel_paths
        .par_iter()
        .map(|rel| {
            let full = root.join(rel);
            match std::fs::read(&full) {
                Ok(content) => {
                    // Skip binary content (heuristic).
                    if content.contains(&0u8) {
                        None
                    } else {
                        Some(file_ngram_hashes(&content, &wf))
                    }
                }
                Err(e) => {
                    eprintln!("warning: skip {:?}: {e}", full);
                    None
                }
            }
        })
        .collect();

    // ── Phase 3: merge into the inverted index (parallel chunk merge) ───────
    skipped += per_file.iter().filter(|h| h.is_none()).count();
    *builder = IndexBuilder::from_file_hashes(&rel_paths, per_file, default_weight_fn());
    let file_count = builder.file_count();

    if file_count == 0 {
        eprintln!("warning: no files found to index");
    } else {
        eprintln!("indexed {} files ({} skipped)", file_count, skipped);
    }

    Ok(())
}

fn is_likely_binary(path: &Path) -> bool {
    // FN-4: Keep this list in sync with `is_likely_binary` in
    // sparse-ngram-git (crates/git/src/git_index.rs) — binary formats only.
    // Text types (e.g. lock, svg, toml) must stay indexable; the NUL-byte
    // content check during indexing is the real binary fallback.
    let binary_exts = [
        "png", "jpg", "jpeg", "gif", "bmp", "ico", "pdf", "zip", "tar", "gz", "bz2", "xz", "7z",
        "rar", "exe", "dll", "so", "dylib", "a", "o", "wasm", "bin", "dat", "mp3", "mp4", "wav",
        "ogg", "flac", "ttf", "woff", "woff2", "eot", "pyc", "class",
    ];
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        binary_exts.contains(&ext.to_lowercase().as_str())
    } else {
        false
    }
}
