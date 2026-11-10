//! End-to-end CLI integration tests for the `sngs` binary.
//!
//! Each test writes source files to a temporary directory and drives the real
//! `sngs` binary via `assert_cmd`.  The binary is built automatically by
//! Cargo before tests run.

use std::fs;

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helper: create a temp dir pre-populated with source files
// ---------------------------------------------------------------------------

fn setup_dir(files: &[(&str, &str)]) -> TempDir {
    let tmp = TempDir::new().expect("create tempdir");
    for (name, content) in files {
        let path = tmp.path().join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, content).unwrap();
    }
    tmp
}

fn sngs() -> Command {
    Command::cargo_bin("sngs").expect("sngs binary must be available")
}

fn setup_git_dir(files: &[(&str, &str)]) -> TempDir {
    let tmp = setup_dir(files);
    for args in [
        vec!["init", "-q"],
        vec!["add", "."],
        vec![
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-m",
            "initial fixture",
        ],
    ] {
        let output = std::process::Command::new("git")
            .args(&args)
            .current_dir(tmp.path())
            .output()
            .expect("run git");
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    tmp
}

fn external_sngs(index_dir: &std::path::Path, via_env: bool) -> Command {
    let mut command = sngs();
    command.env("SNGS_NO_STATS", "1");
    if via_env {
        command.env("SNGS_INDEX_DIR", index_dir);
    } else {
        command
            .env_remove("SNGS_INDEX_DIR")
            .arg("--index-dir")
            .arg(index_dir);
    }
    command
}

fn assert_external_index_preserves_worktree(
    root: &std::path::Path,
    index_dir: &std::path::Path,
    gitignore: Option<&str>,
) {
    assert!(!root.join(".sngi-store").exists());
    match gitignore {
        Some(content) => assert_eq!(
            fs::read(root.join(".gitignore")).unwrap(),
            content.as_bytes()
        ),
        None => assert!(!root.join(".gitignore").exists()),
    }
    let stores: Vec<_> = fs::read_dir(index_dir)
        .unwrap()
        .map(|entry| entry.unwrap().path().join(".sngi-store"))
        .collect();
    assert_eq!(stores.len(), 1);
    assert!(stores[0].join("state.json").exists());
    assert!(stores[0].join("head.txt").exists());

    let status = std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=all"])
        .current_dir(root)
        .output()
        .unwrap();
    assert!(status.status.success());
    assert!(
        status.stdout.is_empty(),
        "workspace changed: {}",
        String::from_utf8_lossy(&status.stdout)
    );
}

// External indexes must never create or edit repository ignore rules, whether
// selected by the CLI or the environment, or created explicitly or by search.
#[test]
fn cli_external_index_init_preserves_worktree() {
    for via_env in [false, true] {
        for gitignore in [None, Some("# existing rules\n/build")] {
            let mut files = vec![("source.rs", "fn external_index_marker() {}\n")];
            if let Some(content) = gitignore {
                files.push((".gitignore", content));
            }
            let tmp = setup_git_dir(&files);
            let cache = TempDir::new().unwrap();
            let path = tmp.path().to_str().unwrap();

            for command in ["init", "init", "sync", "rebuild"] {
                external_sngs(cache.path(), via_env)
                    .args([command, "--path", path])
                    .assert()
                    .success();
                assert_external_index_preserves_worktree(tmp.path(), cache.path(), gitignore);
            }
            for _ in 0..2 {
                external_sngs(cache.path(), via_env)
                    .args(["search", "external_index_marker", "-l", "--path", path])
                    .assert()
                    .success()
                    .stdout("source.rs\n");
                assert_external_index_preserves_worktree(tmp.path(), cache.path(), gitignore);
            }
        }
    }
}

#[test]
fn cli_external_index_auto_init_preserves_worktree() {
    for via_env in [false, true] {
        for explicit_auto_init in [false, true] {
            for gitignore in [None, Some("# existing rules\n/build")] {
                let mut files = vec![("source.rs", "fn external_index_marker() {}\n")];
                if let Some(content) = gitignore {
                    files.push((".gitignore", content));
                }
                let tmp = setup_git_dir(&files);
                let cache = TempDir::new().unwrap();
                let mut command = external_sngs(cache.path(), via_env);
                command.args([
                    "search",
                    "external_index_marker",
                    "-l",
                    "--path",
                    tmp.path().to_str().unwrap(),
                ]);
                if explicit_auto_init {
                    command.arg("--auto-init");
                }
                command
                    .assert()
                    .success()
                    .stdout(predicate::str::contains("source.rs\n"));
                assert_external_index_preserves_worktree(tmp.path(), cache.path(), gitignore);
            }
        }
    }
}

#[test]
fn cli_default_index_init_still_ignores_store() {
    let tmp = setup_git_dir(&[
        ("source.rs", "fn local_index_marker() {}\n"),
        (".gitignore", "# existing rules\n/build"),
    ]);
    for _ in 0..2 {
        sngs()
            .env_remove("SNGS_INDEX_DIR")
            .env("SNGS_NO_STATS", "1")
            .args(["init", "--path", tmp.path().to_str().unwrap()])
            .assert()
            .success();
        assert_eq!(
            fs::read_to_string(tmp.path().join(".gitignore")).unwrap(),
            "# existing rules\n/build\n.sngi-store/\n"
        );
        assert!(tmp.path().join(".sngi-store/state.json").exists());
    }
}

#[cfg(unix)]
#[test]
fn cli_external_index_searches_readonly_git_repository() {
    use std::os::unix::fs::PermissionsExt;

    let tmp = setup_git_dir(&[
        ("source.rs", "fn external_index_marker() {}\n"),
        (".gitignore", "# existing rules\n/build"),
    ]);
    let cache = TempDir::new().unwrap();
    let root_permissions = fs::metadata(tmp.path()).unwrap().permissions();
    let ignore_path = tmp.path().join(".gitignore");
    let ignore_permissions = fs::metadata(&ignore_path).unwrap().permissions();
    fs::set_permissions(&ignore_path, fs::Permissions::from_mode(0o444)).unwrap();
    fs::set_permissions(tmp.path(), fs::Permissions::from_mode(0o555)).unwrap();

    // Restore permissions before asserting so TempDir can always clean up.
    let output = external_sngs(cache.path(), false)
        .args([
            "search",
            "external_index_marker",
            "--auto-init",
            "-l",
            "--path",
            tmp.path().to_str().unwrap(),
        ])
        .output();
    fs::set_permissions(tmp.path(), root_permissions).unwrap();
    fs::set_permissions(ignore_path, ignore_permissions).unwrap();

    let output = output.unwrap();
    assert!(
        output.status.success(),
        "search failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("source.rs\n"));
    assert_external_index_preserves_worktree(
        tmp.path(),
        cache.path(),
        Some("# existing rules\n/build"),
    );
}

// ---------------------------------------------------------------------------
// 1. init creates index; status shows file count
// ---------------------------------------------------------------------------

#[test]
fn cli_init_then_status() {
    let tmp = setup_dir(&[
        ("main.rs", "fn main() { println!(\"hello\"); }"),
        ("lib.rs", "pub fn add(a: i32, b: i32) -> i32 { a + b }"),
    ]);

    // init
    sngs()
        .args(["init", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success();

    // status should mention Files and N-gram tokens
    sngs()
        .args(["status", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("Files"))
        .stdout(predicate::str::contains("N-gram tokens"));
}

// ---------------------------------------------------------------------------
// 2. search literal — hit
// ---------------------------------------------------------------------------

#[test]
fn cli_search_literal_hit() {
    let tmp = setup_dir(&[
        (
            "main.rs",
            "fn main() { println!(\"cli_unique_token_aaaa\"); }",
        ),
        ("other.rs", "fn other() {}"),
    ]);

    sngs()
        .args(["init", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success();

    sngs()
        .args([
            "search",
            "cli_unique_token_aaaa",
            "--path",
            tmp.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("main.rs"));
}

// ---------------------------------------------------------------------------
// 3. search literal — miss
// ---------------------------------------------------------------------------

#[test]
fn cli_search_literal_miss() {
    let tmp = setup_dir(&[("a.rs", "fn foo() {}")]);

    sngs()
        .args(["init", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success();

    // T15: No matches returns exit code 1 (ripgrep convention)
    sngs()
        .args([
            "search",
            "xyzzy_never_present_12345",
            "--path",
            tmp.path().to_str().unwrap(),
        ])
        .assert()
        .code(1)
        .stdout(predicate::str::is_empty());
}

// ---------------------------------------------------------------------------
// 4. search regex (-e flag)
// ---------------------------------------------------------------------------

#[test]
fn cli_search_regex() {
    let tmp = setup_dir(&[
        ("types.rs", "struct MyPoint { x: f64, y: f64 }"),
        ("funcs.rs", "fn compute(v: f32) -> f32 { v }"),
    ]);

    sngs()
        .args(["init", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success();

    sngs()
        .args([
            "search",
            r"struct [A-Z]\w+",
            "-e",
            "--path",
            tmp.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("types.rs"));
}

// ---------------------------------------------------------------------------
// 5. -l (files-only) flag suppresses offset info
// ---------------------------------------------------------------------------

#[test]
fn cli_search_files_only() {
    let tmp = setup_dir(&[("files_only.rs", "fn files_only_marker_xyz() {}")]);

    sngs()
        .args(["init", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success();

    let output = sngs()
        .args([
            "search",
            "files_only_marker_xyz",
            "-l",
            "--path",
            tmp.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let stdout = String::from_utf8(output).unwrap();
    // With -l each line is just a path; no "match(es)" text.
    assert!(stdout.contains("files_only.rs"), "should list the file");
    assert!(!stdout.contains("match"), "should not print match count");
}

// ---------------------------------------------------------------------------
// 6. -n limits output count
// ---------------------------------------------------------------------------

#[test]
fn cli_search_max_results() {
    let tmp = setup_dir(&[
        ("a.rs", "fn limit_token_bbb() {}"),
        ("b.rs", "fn limit_token_bbb() {}"),
        ("c.rs", "fn limit_token_bbb() {}"),
    ]);

    sngs()
        .args([
            "init",
            "--no-ignore",
            "--path",
            tmp.path().to_str().unwrap(),
        ])
        .assert()
        .success();

    let output = sngs()
        .args([
            "search",
            "limit_token_bbb",
            "-n",
            "1",
            "--path",
            tmp.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let stdout = String::from_utf8(output).unwrap();
    // Should mention "more results" since there are 3 files but we capped at 1.
    assert!(
        stdout.contains("more results"),
        "should show truncation notice"
    );
}

// ---------------------------------------------------------------------------
// 7. rebuild replaces existing index
// ---------------------------------------------------------------------------

#[test]
fn cli_rebuild() {
    let tmp = setup_dir(&[("rebuild.rs", "fn rebuild_marker_ccc() {}")]);

    sngs()
        .args(["init", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success();
    sngs()
        .args(["rebuild", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success();

    // After rebuild, search still works.
    sngs()
        .args([
            "search",
            "rebuild_marker_ccc",
            "--path",
            tmp.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("rebuild.rs"));
}

// ---------------------------------------------------------------------------
// 8. sync after adding a new file
// ---------------------------------------------------------------------------

#[test]
fn cli_sync_after_new_file() {
    let tmp = setup_dir(&[("base.rs", "fn base() {}")]);

    sngs()
        .args(["init", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success();

    // Verify new token is not yet found.
    // T15: No matches returns exit code 1 (ripgrep convention)
    sngs()
        .args([
            "search",
            "sync_new_unique_ddd",
            "--path",
            tmp.path().to_str().unwrap(),
        ])
        .assert()
        .code(1)
        .stdout(predicate::str::is_empty());

    // Add a new file to the directory.
    fs::write(
        tmp.path().join("new_after_sync.rs"),
        "fn sync_new_unique_ddd() {}",
    )
    .unwrap();

    // Sync (currently does a full rebuild).
    sngs()
        .args(["sync", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success();

    // Now the new file should be discoverable.
    sngs()
        .args([
            "search",
            "sync_new_unique_ddd",
            "--path",
            tmp.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("new_after_sync.rs"));
}

// ---------------------------------------------------------------------------
// 9. search without init → non-zero exit code
// ---------------------------------------------------------------------------

#[test]
fn cli_search_without_init_fails() {
    let tmp = TempDir::new().unwrap();

    sngs()
        .args(["search", "anything", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .failure();
}

// ---------------------------------------------------------------------------
// 10. status without init → non-zero exit code
// ---------------------------------------------------------------------------

#[test]
fn cli_status_without_init_fails() {
    let tmp = TempDir::new().unwrap();

    sngs()
        .args(["status", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .failure();
}

// ---------------------------------------------------------------------------
// 11. double init is idempotent (exits 0, prints warning)
// ---------------------------------------------------------------------------

#[test]
fn cli_double_init_is_idempotent() {
    let tmp = setup_dir(&[("x.rs", "fn x() {}")]);

    sngs()
        .args(["init", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success();

    // Second init should succeed (exit 0) and warn via stderr.
    sngs()
        .args(["init", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success()
        .stderr(predicate::str::contains("already exists"));
}

// ---------------------------------------------------------------------------
// 12. timeout flag — fast command must finish within generous timeout
// ---------------------------------------------------------------------------

#[test]
fn cli_timeout_flag_allows_fast_command() {
    let tmp = setup_dir(&[("t.rs", "fn timeout_test() {}")]);

    sngs()
        .args([
            "--timeout",
            "30",
            "init",
            "--path",
            tmp.path().to_str().unwrap(),
        ])
        .assert()
        .success();
}

// ---------------------------------------------------------------------------
// Chinese / CJK CLI tests
// ---------------------------------------------------------------------------

#[test]
fn cli_search_chinese_literal() {
    let tmp = setup_dir(&[
        ("zh.rs", "// 你好世界\nfn greet() {}"),
        ("other.rs", "fn other() {}"),
    ]);

    sngs()
        .args(["init", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success();

    sngs()
        .args(["search", "你好世界", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("zh.rs"));
}

#[test]
fn cli_search_chinese_regex() {
    let tmp = setup_dir(&[
        ("zh.rs", "// 你好世界\nfn greet() {}"),
        ("other.rs", "fn other() {}"),
    ]);

    sngs()
        .args(["init", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success();

    sngs()
        .args([
            "search",
            "你好.*世界",
            "-e",
            "--path",
            tmp.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("zh.rs"));
}

#[test]
fn cli_search_chinese_files_only() {
    let tmp = setup_dir(&[
        ("zh.rs", "// 你好世界\nfn greet() {}"),
        ("other.rs", "fn other() {}"),
    ]);

    sngs()
        .args(["init", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success();

    let output = sngs()
        .args([
            "search",
            "你好世界",
            "-l",
            "--path",
            tmp.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let stdout = String::from_utf8(output).unwrap();
    assert!(stdout.contains("zh.rs"), "should list zh.rs");
    assert!(!stdout.contains("match"), "should not print match count");
}

#[test]
fn cli_search_chinese_miss() {
    let tmp = setup_dir(&[("zh.rs", "// 你好世界")]);

    sngs()
        .args(["init", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success();

    // T15: No matches returns exit code 1 (ripgrep convention)
    sngs()
        .args(["search", "再见世界", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .code(1)
        .stdout(predicate::str::is_empty());
}

// ---------------------------------------------------------------------------
// Line numbers and context CLI tests
// ---------------------------------------------------------------------------

#[test]
fn cli_search_with_line_numbers() {
    let tmp = setup_dir(&[("src.rs", "fn foo() {}\nfn bar() {}\nfn baz() {}")]);

    sngs()
        .args(["init", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success();

    // --line-numbers implies context=0 but still prints file:line:content format.
    let output = sngs()
        .args([
            "search",
            "fn bar",
            "--line-numbers",
            "--path",
            tmp.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let stdout = String::from_utf8(output).unwrap();
    // Should contain "src.rs:2:" (line 2 is "fn bar() {}").
    assert!(
        stdout.contains("src.rs:2:"),
        "should output file:line: format, got: {stdout}"
    );
}

#[test]
fn cli_search_with_context() {
    let tmp = setup_dir(&[("src.rs", "line1\nline2\nfn target() {}\nline4\nline5")]);

    sngs()
        .args(["init", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success();

    let output = sngs()
        .args([
            "search",
            "fn target",
            "-C",
            "1",
            "--path",
            tmp.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let stdout = String::from_utf8(output).unwrap();
    // Context line before the match (line2) and after (line4) should appear.
    assert!(
        stdout.contains("line2"),
        "context before should appear, got: {stdout}"
    );
    assert!(
        stdout.contains("fn target"),
        "match line should appear, got: {stdout}"
    );
    assert!(
        stdout.contains("line4"),
        "context after should appear, got: {stdout}"
    );
}

#[test]
fn cli_search_context_with_separator() {
    // Two non-adjacent matches → should have "--" separator between them.
    let content = "fn alpha() {}\nblank\nblank\nblank\nfn alpha() {}";
    let tmp = setup_dir(&[("src.rs", content)]);

    sngs()
        .args(["init", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success();

    let output = sngs()
        .args([
            "search",
            "fn alpha",
            "-C",
            "0",
            "--path",
            tmp.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let stdout = String::from_utf8(output).unwrap();
    // With context=0, each match is a single line; two non-adjacent matches get a "--" separator.
    assert!(
        stdout.contains("--"),
        "separator should appear, got: {stdout}"
    );
}

#[test]
fn cli_search_line_numbers_with_chinese() {
    let tmp = setup_dir(&[("zh.rs", "// comment\n// 你好世界\nfn greet() {}")]);

    sngs()
        .args(["init", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success();

    let output = sngs()
        .args([
            "search",
            "你好世界",
            "--line-numbers",
            "--path",
            tmp.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let stdout = String::from_utf8(output).unwrap();
    // "你好世界" is on line 2.
    assert!(
        stdout.contains("zh.rs:2:"),
        "should output zh.rs:2: for Chinese content on line 2, got: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// Usability-focused CLI tests
// ---------------------------------------------------------------------------

#[test]
fn cli_search_auto_init_builds_missing_index() {
    let tmp = setup_dir(&[("auto.rs", "fn auto_init_marker() {}")]);

    sngs()
        .args([
            "search",
            "auto_init_marker",
            "--auto-init",
            "--path",
            tmp.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("auto.rs"));
}

#[test]
fn cli_search_json_output() {
    let tmp = setup_dir(&[("json.rs", "fn json_marker() {}\n")]);

    sngs()
        .args(["init", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success();

    let output = sngs()
        .args([
            "search",
            "json_marker",
            "--json",
            "--path",
            tmp.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let parsed: serde_json::Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(parsed[0]["path"], "json.rs");
    assert_eq!(parsed[0]["match_count"], 1);
}

#[test]
fn cli_search_count_and_path_filters() {
    let tmp = setup_dir(&[
        (
            "src/keep.rs",
            "fn filter_marker() {}\nfn filter_marker() {}",
        ),
        ("tests/drop.rs", "fn filter_marker() {}"),
        ("src/other.py", "def filter_marker(): pass"),
    ]);

    sngs()
        .args(["init", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success();

    let output = sngs()
        .args([
            "search",
            "filter_marker",
            "--count",
            "--glob",
            "src/*",
            "--type",
            "rs",
            "--path",
            tmp.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let stdout = String::from_utf8(output).unwrap();
    assert!(stdout.contains("src/keep.rs:2"), "got: {stdout}");
    assert!(!stdout.contains("tests/drop.rs"), "got: {stdout}");
    assert!(!stdout.contains("src/other.py"), "got: {stdout}");
}

// ---------------------------------------------------------------------------
// FN-4 regression: .lock files are text and must be indexed/searched
// ---------------------------------------------------------------------------

#[test]
fn cli_search_indexes_lock_extension_files() {
    let tmp = setup_dir(&[
        ("main.rs", "fn main() {}"),
        (
            "Cargo.lock",
            "# This file is automatically @generated by Cargo.\nversion = 4\n\n[[package]]\nname = \"lock_marker_unique_pkg\"\nversion = \"0.1.0\"\n",
        ),
    ]);

    sngs()
        .args(["init", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success();

    // Before the FN-4 fix, `lock` was on the binary-extension blacklist, so
    // Cargo.lock never entered the index and this search found nothing.
    sngs()
        .args([
            "search",
            "lock_marker_unique_pkg",
            "--path",
            tmp.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Cargo.lock"));
}

#[test]
fn cli_doctor_and_warmup() {
    let tmp = setup_dir(&[("doctor.rs", "fn doctor_marker() {}")]);

    sngs()
        .args(["init", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success();

    sngs()
        .args(["doctor", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("Index exists    : true"));

    sngs()
        .args(["warmup", "--path", tmp.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("Warmed"));
}

// ---------------------------------------------------------------------------
// Rebuild inside a git repository must keep the git-aware path
// ---------------------------------------------------------------------------

/// Regression: `sngs rebuild` in a git repo used to fall back to a
/// filesystem walk with .gitignore rules, silently dropping tracked-but-
/// ignored files (e.g. a force-added file under an ignored directory) and
/// destroying the git-aware metadata (head.txt).
#[test]
fn cli_rebuild_in_git_repo_keeps_tracked_but_ignored_files() {
    if std::process::Command::new("git")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("git not available; skipping");
        return;
    }

    let tmp = setup_dir(&[
        (".gitignore", "/ignored_dir\n"),
        ("ignored_dir/keeper.txt", "rebuild_git_marker_xq1 content\n"),
        ("normal.rs", "fn normal() {}\n"),
    ]);
    let root = tmp.path();

    let git = |args: &[&str]| {
        let out = std::process::Command::new("git")
            .args(args)
            .current_dir(root)
            .output()
            .expect("run git");
        assert!(
            out.status.success(),
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr)
        );
    };
    git(&["init", "-q"]);
    git(&[
        "-c",
        "user.email=t@t",
        "-c",
        "user.name=t",
        "add",
        "-f",
        ".",
    ]);
    git(&[
        "-c",
        "user.email=t@t",
        "-c",
        "user.name=t",
        "commit",
        "-q",
        "-m",
        "init",
    ]);

    // Both init and rebuild must keep the force-added file searchable.
    for cmd in ["init", "rebuild"] {
        sngs()
            .args([cmd, "--path", root.to_str().unwrap()])
            .assert()
            .success();
        sngs()
            .args([
                "search",
                "-l",
                "rebuild_git_marker_xq1",
                "--path",
                root.to_str().unwrap(),
            ])
            .assert()
            .success()
            .stdout(predicate::str::contains("ignored_dir/keeper.txt"));
        // The git-aware metadata must survive a rebuild.
        assert!(
            root.join(".sngi-store/head.txt").exists(),
            "{cmd}: head.txt missing — rebuild destroyed git-aware metadata"
        );
    }
}
