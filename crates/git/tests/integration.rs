//! End-to-end integration tests for `sparse-ngram-git`.
//!
//! Each test creates a real temporary git repository (via `git2`), commits
//! files, builds a `GitAwareIndex`, then modifies the working tree and calls
//! `sync()` to exercise the overlay pipeline.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use git2::{Repository, Signature};
use sparse_ngram_git::GitAwareIndex;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Git helpers
// ---------------------------------------------------------------------------

/// Initialise a new bare-ish repo, set a user identity, and create the first
/// commit with the given files.  Returns `(TempDir, Repository)`.
fn init_repo_with_files(files: &[(&str, &[u8])]) -> Result<(TempDir, Repository)> {
    let tmp = TempDir::new()?;
    let repo = Repository::init(tmp.path())?;

    // git2 requires a user identity for commits.
    {
        let mut cfg = repo.config()?;
        cfg.set_str("user.name", "Test User")?;
        cfg.set_str("user.email", "test@example.com")?;
    }

    commit_files(&repo, tmp.path(), files, "initial commit")?;
    Ok((tmp, repo))
}

/// Write `files` to disk, stage them, and create a commit on the current
/// branch (or the initial commit if the repo is empty).
fn commit_files(
    repo: &Repository,
    root: &Path,
    files: &[(&str, &[u8])],
    message: &str,
) -> Result<git2::Oid> {
    for (rel, content) in files {
        let full = root.join(rel);
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&full, content)?;
    }

    let mut index = repo.index()?;
    index.add_all(["*"].iter(), git2::IndexAddOption::DEFAULT, None)?;
    index.write()?;
    let tree_oid = index.write_tree()?;
    let tree = repo.find_tree(tree_oid)?;

    let sig = Signature::now("Test User", "test@example.com")?;
    let oid = if repo.head().is_err() {
        // Initial commit — no parent.
        repo.commit(Some("HEAD"), &sig, &sig, message, &tree, &[])?
    } else {
        let parent_commit = repo.head()?.peel_to_commit()?;
        repo.commit(Some("HEAD"), &sig, &sig, message, &tree, &[&parent_commit])?
    };
    Ok(oid)
}

// ---------------------------------------------------------------------------
// 1. Build from HEAD and search — basic smoke test
// ---------------------------------------------------------------------------

#[test]
fn build_and_search_literal() -> Result<()> {
    let (tmp, _repo) = init_repo_with_files(&[
        ("main.rs", b"fn main() { println!(\"hello_git_test\"); }"),
        ("lib.rs", b"pub fn add(a: i32, b: i32) -> i32 { a + b }"),
    ])?;

    let gi = GitAwareIndex::open_or_build(tmp.path())?;
    let results = gi.search("hello_git_test", None)?;

    assert_eq!(results.len(), 1, "only main.rs contains hello_git_test");
    assert_eq!(results[0].path, PathBuf::from("main.rs"));
    Ok(())
}

// ---------------------------------------------------------------------------
// 2. Literal pattern not present → empty results
// ---------------------------------------------------------------------------

#[test]
fn search_miss_returns_empty() -> Result<()> {
    let (tmp, _repo) = init_repo_with_files(&[("a.rs", b"fn foo() {}"), ("b.rs", b"fn bar() {}")])?;

    let gi = GitAwareIndex::open_or_build(tmp.path())?;
    let results = gi.search("xyzzy_never_present_99999", None)?;
    assert!(results.is_empty());
    Ok(())
}

// ---------------------------------------------------------------------------
// 3. Overlay: uncommitted new file is discoverable after sync
// ---------------------------------------------------------------------------

#[test]
fn overlay_new_untracked_file_found_after_sync() -> Result<()> {
    let (tmp, repo) = init_repo_with_files(&[("existing.rs", b"fn existing() {}")])?;

    // Build base index.
    let mut gi = GitAwareIndex::open_or_build(tmp.path())?;

    // Write a new file to the working tree but do NOT commit it.
    let new_path = tmp.path().join("new_overlay.rs");
    fs::write(&new_path, b"fn overlay_unique_marker_abc() {}")?;

    // Stage the file so git2's workdir diff picks it up.
    let mut index = repo.index()?;
    index.add_path(Path::new("new_overlay.rs"))?;
    index.write()?;

    // Sync should pick up the dirty file into the overlay.
    let stats = gi.sync()?;
    assert!(
        stats.modified > 0 || stats.added > 0,
        "sync should detect new file"
    );

    let results = gi.search("overlay_unique_marker_abc", None)?;
    assert_eq!(results.len(), 1, "new overlay file must be searchable");
    Ok(())
}

// ---------------------------------------------------------------------------
// 4. Overlay: deleted file is hidden from results after sync
// ---------------------------------------------------------------------------

#[test]
fn overlay_deleted_file_hidden_after_sync() -> Result<()> {
    let (tmp, repo) = init_repo_with_files(&[
        ("keep.rs", b"fn keep() { let k = 1; }"),
        ("delete.rs", b"fn delete_me_unique_xyz() {}"),
    ])?;

    let mut gi = GitAwareIndex::open_or_build(tmp.path())?;

    // Verify it's found initially.
    let before = gi.search("delete_me_unique_xyz", None)?;
    assert_eq!(before.len(), 1, "should be found before deletion");

    // Delete from working tree and stage.
    fs::remove_file(tmp.path().join("delete.rs"))?;
    let mut index = repo.index()?;
    index.remove_path(Path::new("delete.rs"))?;
    index.write()?;

    let stats = gi.sync()?;
    assert!(stats.deleted > 0, "sync should detect deletion");

    let after = gi.search("delete_me_unique_xyz", None)?;
    assert!(after.is_empty(), "deleted file must not appear in results");
    Ok(())
}

// ---------------------------------------------------------------------------
// 5. Overlay: modified file content is reflected after sync
// ---------------------------------------------------------------------------

#[test]
fn overlay_modified_file_reflects_new_content() -> Result<()> {
    let (tmp, repo) = init_repo_with_files(&[("mod.rs", b"fn original_token_aaa() {}")])?;

    let mut gi = GitAwareIndex::open_or_build(tmp.path())?;

    // Overwrite the file with new content.
    let file_path = tmp.path().join("mod.rs");
    fs::write(&file_path, b"fn replaced_token_bbb() {}")?;

    // Stage the modification.
    let mut index = repo.index()?;
    index.add_path(Path::new("mod.rs"))?;
    index.write()?;

    gi.sync()?;

    // Old token should no longer be found (shadowed by overlay).
    let old_results = gi.search("original_token_aaa", None)?;
    assert!(
        old_results.is_empty(),
        "old token must be shadowed by overlay"
    );

    // New token should be found via overlay.
    let new_results = gi.search("replaced_token_bbb", None)?;
    assert_eq!(new_results.len(), 1, "new token must be found via overlay");
    Ok(())
}

// ---------------------------------------------------------------------------
// 6. New commit advances HEAD — diff picked up by sync
// ---------------------------------------------------------------------------

#[test]
fn new_commit_picked_up_by_sync() -> Result<()> {
    let (tmp, repo) = init_repo_with_files(&[("base.rs", b"fn base() {}")])?;

    let mut gi = GitAwareIndex::open_or_build(tmp.path())?;

    // Add a new commit with an additional file.
    commit_files(
        &repo,
        tmp.path(),
        &[("committed_new.rs", b"fn committed_unique_qqq() {}")],
        "add committed_new.rs",
    )?;

    gi.sync()?;

    let results = gi.search("committed_unique_qqq", None)?;
    assert_eq!(
        results.len(),
        1,
        "newly committed file must be searchable after sync"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 7. open_or_build is idempotent — second call reuses existing index
// ---------------------------------------------------------------------------

#[test]
fn open_or_build_is_idempotent() -> Result<()> {
    let (tmp, _repo) = init_repo_with_files(&[("idem.rs", b"fn idempotent_token_zzz() {}")])?;

    // First open: builds from scratch.
    let gi1 = GitAwareIndex::open_or_build(tmp.path())?;
    let commit1 = gi1.base_commit().to_string();

    // Second open: reuses index.
    let gi2 = GitAwareIndex::open_or_build(tmp.path())?;
    assert_eq!(
        gi2.base_commit(),
        commit1,
        "base commit must be stable across opens"
    );

    let results = gi2.search("idempotent_token_zzz", None)?;
    assert_eq!(results.len(), 1);
    Ok(())
}

// ---------------------------------------------------------------------------
// 8. Regex search through GitAwareIndex
// ---------------------------------------------------------------------------

#[test]
fn regex_search_through_git_index() -> Result<()> {
    let (tmp, _repo) = init_repo_with_files(&[
        ("regex_a.rs", b"struct Alpha { x: u32 }"),
        ("regex_b.rs", b"fn beta() -> u32 { 42 }"),
        ("regex_c.rs", b"let gamma = \"hello\";"),
    ])?;

    let gi = GitAwareIndex::open_or_build(tmp.path())?;
    let results = gi.search(r"struct [A-Z]\w+", None)?;

    assert!(
        results.iter().any(|r| r.path == Path::new("regex_a.rs")),
        "regex_a.rs must match struct [A-Z]\\w+"
    );
    assert!(
        !results.iter().any(|r| r.path == Path::new("regex_b.rs")),
        "regex_b.rs should not match struct pattern"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 9. Dirty-scan cache: sync_if_stale reuses a fresh scan within the TTL
// ---------------------------------------------------------------------------

#[test]
fn dirty_scan_cache_hit_within_ttl() -> Result<()> {
    let (tmp, _repo) = init_repo_with_files(&[("cached.rs", b"fn cached_token_aa11() {}")])?;

    let mut gi = GitAwareIndex::open_or_build(tmp.path())?;
    // Pin a huge TTL so the cache never expires during this test.
    gi.set_dirty_scan_ttl(std::time::Duration::from_secs(86_400));

    // Establish the scan cache with a forced full sync.
    gi.sync()?;
    let cache_file = tmp.path().join(".sngi-store").join("dirty_scan.json");
    assert!(
        cache_file.exists(),
        "forced sync must persist the scan cache"
    );

    // A file created AFTER the cached scan is invisible to sync_if_stale
    // (accepted staleness within the TTL window) but visible to a forced sync.
    fs::write(tmp.path().join("late.rs"), b"fn late_token_bb22() {}")?;

    gi.sync_if_stale()?;
    assert!(
        gi.search("late_token_bb22", None)?.is_empty(),
        "late file is intentionally invisible within the TTL window \
         (proves the cached scan was reused instead of rescanning)"
    );

    gi.sync()?;
    assert_eq!(
        gi.search("late_token_bb22", None)?.len(),
        1,
        "forced sync must discover the late file"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 10. Dirty-scan cache: TTL=0 disables reuse (always rescan)
// ---------------------------------------------------------------------------

#[test]
fn dirty_scan_cache_disabled_with_zero_ttl() -> Result<()> {
    let (tmp, _repo) = init_repo_with_files(&[("zero.rs", b"fn zero_token_cc33() {}")])?;

    let mut gi = GitAwareIndex::open_or_build(tmp.path())?;
    gi.set_dirty_scan_ttl(std::time::Duration::ZERO);

    gi.sync()?;
    fs::write(tmp.path().join("fresh.rs"), b"fn fresh_token_dd44() {}")?;

    let stats = gi.sync_if_stale()?;
    assert!(stats.modified > 0, "TTL=0 must always rescan");
    assert_eq!(gi.search("fresh_token_dd44", None)?.len(), 1);
    Ok(())
}

// ---------------------------------------------------------------------------
// 11. Dirty-scan cache: a cached dirty file still tracks current content,
//     and its deletion is picked up via revalidation
// ---------------------------------------------------------------------------

#[test]
fn dirty_scan_cache_hit_revalidates_listed_files() -> Result<()> {
    let (tmp, _repo) = init_repo_with_files(&[("track.rs", b"fn track_v1_ee55() {}")])?;

    let mut gi = GitAwareIndex::open_or_build(tmp.path())?;
    gi.set_dirty_scan_ttl(std::time::Duration::from_secs(86_400));

    // Make the file dirty, then force a scan so it enters the cache.
    fs::write(tmp.path().join("track.rs"), b"fn track_v2_ff66() {}")?;
    gi.sync()?;
    assert_eq!(gi.search("track_v2_ff66", None)?.len(), 1);

    // Modify it again — the cached list still contains the path, and a
    // cache-hit sync must re-read its CURRENT content.
    fs::write(tmp.path().join("track.rs"), b"fn track_v3_gg77() {}")?;
    gi.sync_if_stale()?;
    assert_eq!(
        gi.search("track_v3_gg77", None)?.len(),
        1,
        "cached dirty file must reflect current content"
    );
    assert!(gi.search("track_v1_ee55", None)?.is_empty());

    // Delete it — a cache-hit sync must hide stale base results.
    fs::remove_file(tmp.path().join("track.rs"))?;
    gi.sync_if_stale()?;
    assert!(
        gi.search("track_v3_gg77", None)?.is_empty(),
        "deleted cached file must be hidden after revalidation"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 12. Dirty-scan cache: a commit invalidates the cached scan
// ---------------------------------------------------------------------------

#[test]
fn dirty_scan_cache_invalidated_by_new_commit() -> Result<()> {
    let (tmp, repo) = init_repo_with_files(&[("inv.rs", b"fn inv_token_hh88() {}")])?;

    let mut gi = GitAwareIndex::open_or_build(tmp.path())?;
    gi.set_dirty_scan_ttl(std::time::Duration::from_secs(86_400));
    gi.sync()?;

    // New commit changes HEAD → cached scan (taken against the old commit)
    // must be discarded even within the TTL window.
    commit_files(
        &repo,
        tmp.path(),
        &[("inv2.rs", b"fn inv2_token_ii99() {}")],
        "second commit",
    )?;

    gi.sync_if_stale()?;
    assert_eq!(
        gi.search("inv2_token_ii99", None)?.len(),
        1,
        "new commit must be visible despite a fresh cache"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 13. FP-2 regression: empty-matching regex must not resurrect deleted files
// ---------------------------------------------------------------------------

#[test]
fn overlay_deleted_file_hidden_for_empty_matching_regex() -> Result<()> {
    let (tmp, repo) = init_repo_with_files(&[
        ("keep.rs", b"fn keep() { let k = 1; }\n"),
        ("gone.rs", b"fn gone_unique_pp11() {}\n"),
    ])?;

    let mut gi = GitAwareIndex::open_or_build(tmp.path())?;

    fs::remove_file(tmp.path().join("gone.rs"))?;
    let mut index = repo.index()?;
    index.remove_path(Path::new("gone.rs"))?;
    index.write()?;

    gi.sync()?;

    // `.*` matches any content — including the empty byte string the overlay
    // masking trick used to feed the verifier — so it is the sharpest probe
    // for deleted files leaking into results.
    let results = gi.search(".*", None)?;
    assert!(
        !results.iter().any(|r| r.path == Path::new("gone.rs")),
        "deleted file must not appear even for empty-matching regex"
    );
    assert!(
        results.iter().any(|r| r.path == Path::new("keep.rs")),
        "live file must still match .*"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 14. FP-2 regression: modified file reports offsets from current content
// ---------------------------------------------------------------------------

#[test]
fn overlay_modified_file_reports_current_offsets() -> Result<()> {
    let (tmp, repo) = init_repo_with_files(&[("mod.rs", b"fn original_token_qq22() {}\n")])?;

    let mut gi = GitAwareIndex::open_or_build(tmp.path())?;

    let new_content: &[u8] = b"hello world\nsecond line here\n";
    fs::write(tmp.path().join("mod.rs"), new_content)?;
    let mut index = repo.index()?;
    index.add_path(Path::new("mod.rs"))?;
    index.write()?;

    gi.sync()?;

    // An empty-matching regex used to let the stale store entry shadow the
    // overlay result with bogus offsets.
    let results = gi.search(".*", None)?;
    let matches: Vec<_> = results
        .iter()
        .filter(|r| r.path == Path::new("mod.rs"))
        .collect();
    assert_eq!(matches.len(), 1, "mod.rs must appear exactly once");

    let re = regex::bytes::Regex::new(".*")?;
    let want: Vec<usize> = re.find_iter(new_content).map(|m| m.start()).collect();
    assert_eq!(
        matches[0].match_offsets, want,
        "offsets must come from the current working-tree content"
    );
    Ok(())
}
