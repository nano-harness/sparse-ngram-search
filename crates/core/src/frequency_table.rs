//! Byte-pair frequency table and inverse-frequency weight function.
//!
//! The `FrequencyWeight` struct maps each byte pair `(a, b)` to a weight
//! proportional to how *rare* the pair is in typical source code.  Rare pairs
//! produce high weights and therefore become preferred n-gram boundaries.
//!
//! The built-in table (`DEFAULT_FREQUENCY_TABLE`) is a 256×256 matrix of
//! `u32` weights stored as a flat array in row-major order (row = first byte,
//! col = second byte).  It is embedded at compile time via `include_bytes!`.
//!
//! The weights were derived by:
//!   1. Counting byte-pair frequencies across a large corpus of open-source
//!      code (see the `freq-gen` crate for the generator tool).
//!   2. Inverting the frequency: `weight = MAX_FREQ / (count + 1)` so that
//!      pairs that never appear get the maximum weight.

use std::path::Path;
use std::sync::{Arc, OnceLock};

use anyhow::Result;

use crate::ngram::WeightFn;

// ---------------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------------

/// Returns `true` if `b` is a UTF-8 continuation byte (`10xxxxxx`, i.e. `0x80–0xBF`).
#[inline(always)]
fn is_continuation_byte(b: u8) -> bool {
    (b & 0xC0) == 0x80
}

// ---------------------------------------------------------------------------
// Built-in frequency table
// ---------------------------------------------------------------------------

/// The built-in 256×256 weight table, embedded at compile time.
///
/// Layout: `TABLE[a as usize * 256 + b as usize]` = weight of pair (a, b).
///
/// We store `u16` values (0..=65535) packed as little-endian bytes to keep
/// the binary size manageable (128 KiB).
static BUILTIN_TABLE_BYTES: &[u8] = include_bytes!("../data/frequency_table.bin");

// T24: Cache parsed builtin table with OnceLock
static BUILTIN_TABLE_CACHE: OnceLock<Arc<[u32; 65536]>> = OnceLock::new();

// The UTF-8-smoothed variant is the default weight function; it is built on
// every `default_weight_fn()` call, so it is cached too (a parse + smoothing
// pass costs ~100µs and allocates 256 KiB — noticeable per segment per query).
static SMOOTHED_TABLE_CACHE: OnceLock<Arc<[u32; 65536]>> = OnceLock::new();

/// Parse the built-in table bytes into a boxed flat array.
fn parse_builtin() -> Box<[u32; 65536]> {
    assert_eq!(
        BUILTIN_TABLE_BYTES.len(),
        65536 * 2,
        "frequency_table.bin must be exactly 131072 bytes (65536 u16 LE values)"
    );
    let mut table = Box::new([0u32; 65536]);
    for i in 0..65536 {
        let lo = BUILTIN_TABLE_BYTES[i * 2] as u32;
        let hi = BUILTIN_TABLE_BYTES[i * 2 + 1] as u32;
        table[i] = lo | (hi << 8);
    }
    table
}

// ---------------------------------------------------------------------------
// FrequencyWeight
// ---------------------------------------------------------------------------

/// Weight function backed by a 256×256 byte-pair weight table.
pub struct FrequencyWeight {
    /// Flat 256×256 array; index = `a as usize * 256 + b as usize`.
    table: Arc<[u32; 65536]>,
}

impl FrequencyWeight {
    /// Create a `FrequencyWeight` from the built-in table.
    pub fn builtin() -> Self {
        // T24: Use cached parsed table
        let table = BUILTIN_TABLE_CACHE
            .get_or_init(|| Arc::from(parse_builtin()))
            .clone();
        Self { table }
    }

    /// Create a `FrequencyWeight` from the built-in table with UTF-8 smoothing applied.
    ///
    /// The built-in table was derived from ASCII-heavy source code, so byte pairs in the
    /// high range (`0x80–0xBF`) have very high inverse-frequency weights.  This causes
    /// n-gram splitters to break UTF-8 multi-byte sequences (e.g. Chinese characters) at
    /// *every* continuation byte, producing many tiny, meaningless n-grams.
    ///
    /// This constructor lowers the weight for continuation-byte → continuation-byte pairs
    /// (`0x80–0xBF` × `0x80–0xBF`) to a level comparable to common ASCII pairs so that
    /// whole UTF-8 characters tend to be kept inside the same n-gram instead of being
    /// split at every byte boundary.
    pub fn with_utf8_smoothing() -> Self {
        let table = SMOOTHED_TABLE_CACHE
            .get_or_init(|| {
                let mut table = parse_builtin();
                // Target weight: comparable to common ASCII byte pairs (small = low rarity = don't split).
                let target_weight: u32 = 5;
                for a in 0x80u8..=0xBFu8 {
                    for b in 0x80u8..=0xBFu8 {
                        debug_assert!(is_continuation_byte(a) && is_continuation_byte(b));
                        let idx = a as usize * 256 + b as usize;
                        table[idx] = table[idx].min(target_weight);
                    }
                }
                Arc::from(table)
            })
            .clone();
        Self { table }
    }

    /// Load a custom frequency table from a binary file.
    ///
    /// File format: 65536 little-endian `u16` values (131072 bytes total).
    pub fn from_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        anyhow::ensure!(
            bytes.len() == 65536 * 2,
            "frequency table file must be exactly 131072 bytes, got {}",
            bytes.len()
        );
        let mut table = Box::new([0u32; 65536]);
        for i in 0..65536 {
            let lo = bytes[i * 2] as u32;
            let hi = bytes[i * 2 + 1] as u32;
            table[i] = lo | (hi << 8);
        }
        Ok(Self {
            table: Arc::from(table),
        })
    }

    /// Directly construct from a pre-built weight table (for testing / freq-gen).
    pub fn from_raw(table: Box<[u32; 65536]>) -> Self {
        Self {
            table: Arc::from(table),
        }
    }
}

impl WeightFn for FrequencyWeight {
    #[inline(always)]
    fn weight(&self, a: u8, b: u8) -> u32 {
        self.table[a as usize * 256 + b as usize]
    }
}

// ---------------------------------------------------------------------------
// Convenience constructor
// ---------------------------------------------------------------------------

/// Returns the default (built-in) weight function as a `Box<dyn WeightFn>`.
///
/// The returned table includes UTF-8 continuation-byte smoothing so that
/// whole CJK/multi-byte characters are preferred over splitting at every
/// continuation byte.  Use [`FrequencyWeight::builtin`] if you need the
/// raw, unsmoothed table.
pub fn default_weight_fn() -> Box<dyn WeightFn> {
    Box::new(FrequencyWeight::with_utf8_smoothing())
}

// ---------------------------------------------------------------------------
// Frequency table builder (used by freq-gen and tests)
// ---------------------------------------------------------------------------

/// Accumulates byte-pair counts from a corpus, then converts to weights.
pub struct FrequencyTableBuilder {
    /// Raw occurrence counts.
    counts: Box<[u64; 65536]>,
}

impl FrequencyTableBuilder {
    pub fn new() -> Self {
        Self {
            counts: Box::new([0u64; 65536]),
        }
    }

    /// Feed a chunk of bytes into the counter.
    pub fn feed(&mut self, data: &[u8]) {
        for pair in data.windows(2) {
            let idx = pair[0] as usize * 256 + pair[1] as usize;
            self.counts[idx] = self.counts[idx].saturating_add(1);
        }
    }

    /// Finalise: convert counts to inverse-frequency weights in `[1, 65535]`.
    ///
    /// Formula: `weight = max_weight - scaled_log_freq`
    /// where scaling maps `[0, log(max_count+1)]` → `[0, max_weight-1]`.
    pub fn build_weights(&self) -> Box<[u32; 65536]> {
        let max_count = *self.counts.iter().max().unwrap_or(&0);
        if max_count == 0 {
            // Corpus was empty — use uniform weights.
            let mut t = Box::new([1u32; 65536]);
            // Assign slightly higher weight to non-ASCII pairs so they are
            // preferred as n-gram boundaries over ASCII pairs.
            for a in 0x80u8..=0xFF {
                for b in 0u8..=0xFF {
                    t[a as usize * 256 + b as usize] = 2;
                }
            }
            // Lower the weight for UTF-8 continuation-byte internal pairs
            // (0x80–0xBF → 0x80–0xBF) so whole CJK characters are not split.
            for a in 0x80u8..=0xBFu8 {
                for b in 0x80u8..=0xBFu8 {
                    t[a as usize * 256 + b as usize] = 1;
                }
            }
            return t;
        }

        let max_weight: u32 = 65535;
        let log_max = ((max_count + 1) as f64).ln();

        let mut weights = Box::new([0u32; 65536]);
        for i in 0..65536 {
            let count = self.counts[i];
            let log_freq = ((count + 1) as f64).ln();
            // Scale log_freq / log_max to [0, max_weight-1], then invert.
            let scaled = ((log_freq / log_max) * (max_weight - 1) as f64) as u32;
            weights[i] = (max_weight - scaled).max(1);
        }
        weights
    }

    /// Serialise the weight table to the binary format used by `include_bytes!`.
    pub fn write_to(weights: &[u32; 65536], out: &mut Vec<u8>) {
        for &w in weights.iter() {
            let v = w.min(65535) as u16;
            out.push(v as u8);
            out.push((v >> 8) as u8);
        }
    }
}

impl Default for FrequencyTableBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_table_loads() {
        let wf = FrequencyWeight::builtin();
        // Every weight must be ≥ 1.
        for a in 0u8..=255 {
            for b in 0u8..=255 {
                assert!(wf.weight(a, b) >= 1, "zero weight at ({a}, {b})");
            }
        }
    }

    #[test]
    fn test_builder_empty_corpus() {
        let builder = FrequencyTableBuilder::new();
        let weights = builder.build_weights();
        for &w in weights.iter() {
            assert!(w >= 1);
        }
    }

    #[test]
    fn test_builder_common_vs_rare() {
        let mut builder = FrequencyTableBuilder::new();
        // Feed "aa" 1000× and "zq" once.
        for _ in 0..1000 {
            builder.feed(b"aa");
        }
        builder.feed(b"zq");
        let weights = builder.build_weights();
        let w_common = weights[b'a' as usize * 256 + b'a' as usize];
        let w_rare = weights[b'z' as usize * 256 + b'q' as usize];
        assert!(w_rare > w_common, "rare pair should have higher weight");
    }

    #[test]
    fn test_builder_serialise_roundtrip() {
        let mut builder = FrequencyTableBuilder::new();
        builder.feed(b"hello world, this is a test of the frequency table builder.");
        let weights = builder.build_weights();

        let mut buf = Vec::new();
        FrequencyTableBuilder::write_to(&weights, &mut buf);
        assert_eq!(buf.len(), 65536 * 2);

        // Parse back.
        for i in 0..65536 {
            let lo = buf[i * 2] as u32;
            let hi = buf[i * 2 + 1] as u32;
            let v = lo | (hi << 8);
            assert_eq!(v, weights[i].min(65535));
        }
    }

    #[test]
    fn test_utf8_smoothing_lowers_continuation_weights() {
        let builtin = FrequencyWeight::builtin();
        let smoothed = FrequencyWeight::with_utf8_smoothing();

        // All continuation-byte internal pairs must have weight ≤ 5 after smoothing.
        for a in 0x80u8..=0xBFu8 {
            for b in 0x80u8..=0xBFu8 {
                let w = smoothed.weight(a, b);
                assert!(
                    w <= 5,
                    "continuation pair ({a:#04x}, {b:#04x}) weight {w} must be ≤ 5 after smoothing"
                );
            }
        }

        // Smoothing must not raise any weight that was already low.
        for a in 0x80u8..=0xBFu8 {
            for b in 0x80u8..=0xBFu8 {
                let before = builtin.weight(a, b);
                let after = smoothed.weight(a, b);
                assert!(
                    after <= before,
                    "smoothing must not increase weight for ({a:#04x}, {b:#04x}): {before} → {after}"
                );
            }
        }

        // Smoothing must not change weights outside the continuation-byte range
        // (spot-check a few ASCII and high-byte pairs).
        for a in [b'a', b'z', 0xC0u8, 0xE4u8, 0xFFu8] {
            for b in [b'a', b'z', 0xC0u8, 0xE4u8, 0xFFu8] {
                if is_continuation_byte(a) && is_continuation_byte(b) {
                    continue; // already tested above
                }
                assert_eq!(
                    builtin.weight(a, b),
                    smoothed.weight(a, b),
                    "smoothing must not change weight outside continuation range for ({a:#04x}, {b:#04x})"
                );
            }
        }
    }
}
