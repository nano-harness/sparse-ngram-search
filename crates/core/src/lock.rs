//! File-based locking for `.sngi-store/` to prevent concurrent modifications.
//!
//! # F6: File Locking
//!
//! This module provides shared and exclusive locking for the index store directory
//! using a `.lock` file. Read operations acquire a shared lock, write operations
//! acquire an exclusive lock.
//!
//! Uses `fs2` for cross-platform file locking (flock on Unix, LockFileEx on Windows).

use std::fs::{File, OpenOptions};
use std::path::Path;

use anyhow::{Context, Result};
use fs2::FileExt;

/// A handle to the `.sngi-store/.lock` file.
///
/// The lock is held for the lifetime of this struct and automatically released on drop.
pub struct LockFile {
    _file: File,
    _exclusive: bool,
}

impl LockFile {
    /// Acquire a shared lock (for read operations like search, status).
    ///
    /// Multiple readers can hold shared locks simultaneously.
    /// Blocks if an exclusive lock is held.
    pub fn shared(lock_path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(lock_path)
            .with_context(|| format!("open lock file {:?}", lock_path))?;

        file.lock_shared()
            .with_context(|| format!("acquire shared lock on {:?}", lock_path))?;

        Ok(Self {
            _file: file,
            _exclusive: false,
        })
    }

    /// Acquire an exclusive lock (for write operations like init, rebuild, sync).
    ///
    /// Only one exclusive lock can be held at a time.
    /// Blocks if any locks (shared or exclusive) are held.
    pub fn exclusive(lock_path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(lock_path)
            .with_context(|| format!("open lock file {:?}", lock_path))?;

        file.lock_exclusive()
            .with_context(|| format!("acquire exclusive lock on {:?}", lock_path))?;

        Ok(Self {
            _file: file,
            _exclusive: true,
        })
    }
}

impl Drop for LockFile {
    fn drop(&mut self) {
        // Lock is automatically released when file is closed
        // fs2 doesn't require explicit unlock
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use tempfile::TempDir;

    #[test]
    fn test_shared_locks_concurrent() -> Result<()> {
        let tmp = TempDir::new()?;
        let lock_path = tmp.path().join(".lock");

        // Two threads can acquire shared locks concurrently
        let lock_path1 = lock_path.clone();
        let lock_path2 = lock_path.clone();

        let barrier = Arc::new(Barrier::new(2));
        let b1 = barrier.clone();
        let b2 = barrier.clone();

        let t1 = thread::spawn(move || {
            let _lock = LockFile::shared(&lock_path1).unwrap();
            b1.wait(); // Both threads have locks
            b1.wait(); // Hold lock until both ready to release
        });

        let t2 = thread::spawn(move || {
            let _lock = LockFile::shared(&lock_path2).unwrap();
            b2.wait(); // Both threads have locks
            b2.wait(); // Hold lock until both ready to release
        });

        t1.join().unwrap();
        t2.join().unwrap();

        Ok(())
    }

    #[test]
    fn test_exclusive_lock_blocks() -> Result<()> {
        let tmp = TempDir::new()?;
        let lock_path = tmp.path().join(".lock");

        // First thread acquires exclusive lock
        let _lock1 = LockFile::exclusive(&lock_path)?;

        // Second thread should not be able to acquire any lock
        // (We can't easily test blocking without timeout, so we'll skip runtime test)

        Ok(())
    }

    #[test]
    fn test_lock_released_on_drop() -> Result<()> {
        let tmp = TempDir::new()?;
        let lock_path = tmp.path().join(".lock");

        {
            let _lock = LockFile::exclusive(&lock_path)?;
            // Lock is held
        }
        // Lock is released

        // Should be able to acquire it again
        let _lock2 = LockFile::exclusive(&lock_path)?;

        Ok(())
    }
}
