//! `sparse-ngram-core` – Core library for the sparse n-gram search index.
//!
//! # Crate structure
//!
//! | Module               | Contents                                         |
//! |----------------------|--------------------------------------------------|
//! | [`ngram`]            | `WeightFn` trait, `NGram`, `build_all`, `build_covering` |
//! | [`frequency_table`]  | Built-in byte-pair weight table, `FrequencyTableBuilder` |
//! | [`storage`]          | Binary index file format, `IndexWriter`, `IndexReader` |
//! | [`index`]            | `IndexBuilder`, `Index`, parallel indexing       |
//! | [`regex_decompose`]  | Regex → `QueryAtom` literal tree                 |
//! | [`query`]            | `QueryEngine`, `SearchResult`                    |
//! | [`segment`]          | `SegmentManager`, LSM-tier model                 |
//! | [`compaction`]       | `CompactionEngine`, tiered merge scheduling      |
//! | [`lock`]             | File-based locking for concurrent access         |
//! | [`cancellation`]     | Cooperative cancellation for long operations     |

pub mod cancellation;
pub mod compaction;
pub mod deletion_bitmap;
pub mod frequency_table;
pub mod index;
pub mod lock;
pub mod ngram;
pub mod query;
pub mod regex_decompose;
pub mod segment;
pub mod storage;

// Re-export the most commonly used types at the crate root.
pub use cancellation::CancellationToken;
pub use compaction::{CompactionConfig, CompactionEngine, CompactionPlan, CompactionStats};
pub use deletion_bitmap::{DeletionBitmap, SegmentDeletions};
pub use frequency_table::{default_weight_fn, FrequencyTableBuilder, FrequencyWeight};
pub use index::{file_ngram_hashes, Index, IndexBuilder, IndexMeta};
pub use lock::LockFile;
pub use ngram::{build_all, build_all_hashes, build_covering, fnv_hash, NGram, WeightFn};
pub use query::{
    extract_context_lines, offset_to_match_detail, MatchDetail, QueryEngine, SearchResult,
};
pub use regex_decompose::{decompose_regex, QueryAtom};
pub use segment::{CommitRange, Segment, SegmentLevel, SegmentManager, SegmentStats};
pub use storage::{
    resolve_index_dir, set_external_index_base, IndexPaths, IndexReader, IndexStorePaths,
    IndexWriter, INDEX_STORE_DIR,
};
