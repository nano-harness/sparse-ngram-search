//! Cooperative cancellation support for long-running operations.
//!
//! Provides a simple cancellation token that can be checked periodically
//! during indexing, search, and compaction operations.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A token that can be used to signal cancellation of an operation.
///
/// This is thread-safe and can be cloned to share across tasks.
#[derive(Clone)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
    deadline: Option<Instant>,
}

impl CancellationToken {
    /// Create a new cancellation token with no deadline.
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            deadline: None,
        }
    }

    /// Create a new cancellation token with a timeout.
    pub fn with_timeout(duration: Duration) -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            deadline: Some(Instant::now() + duration),
        }
    }

    /// Check if the operation has been cancelled or timed out.
    pub fn is_cancelled(&self) -> bool {
        if self.cancelled.load(Ordering::Relaxed) {
            return true;
        }
        if let Some(deadline) = self.deadline {
            if Instant::now() >= deadline {
                return true;
            }
        }
        false
    }

    /// Manually cancel the operation.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_cancellation_token_starts_uncancelled() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn test_cancellation_token_can_be_cancelled() {
        let token = CancellationToken::new();
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn test_cancellation_token_with_timeout() {
        let token = CancellationToken::with_timeout(Duration::from_millis(50));
        assert!(!token.is_cancelled());
        thread::sleep(Duration::from_millis(100));
        assert!(token.is_cancelled());
    }

    #[test]
    fn test_cancellation_token_is_cloneable() {
        let token1 = CancellationToken::new();
        let token2 = token1.clone();
        token1.cancel();
        assert!(token2.is_cancelled());
    }
}
