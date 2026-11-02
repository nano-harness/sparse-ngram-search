//! `sparse-ngram-git` – Git integration for sparse-ngram-search.
//!
//! Provides [`GitAwareIndex`] which combines a multi-segment on-disk store
//! (backed by [`sparse_ngram_core::segment::SegmentManager`]) with an
//! in-memory overlay for uncommitted working-tree changes.

pub mod git_index;
pub mod overlay;

pub use git_index::{GitAwareIndex, SyncStats};
pub use overlay::OverlayIndex;
