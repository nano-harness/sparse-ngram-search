//! Sparse N-gram algorithm implementation.
//!
//! The algorithm assigns a weight to each adjacent byte pair in the input.
//! An n-gram boundary is placed wherever both endpoints have strictly greater
//! weight than every interior pair.  This produces substrings that tend to
//! begin and end on "rare" byte pairs (high weight) and contain only "common"
//! pairs (low weight) in the middle — exactly the property we want for a
//! search index: short, distinctive tokens.
//!
//! Two entry points are provided:
//! - `build_all`      – extract every n-gram in the input (for indexing).
//! - `build_covering` – extract the smallest set of n-grams that together
//!   cover the entire input (for query generation).

use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// WeightFn trait
// ---------------------------------------------------------------------------

/// A function that assigns a weight to an adjacent byte pair `(a, b)`.
/// Higher weight ⟹ more "unusual" pair ⟹ preferred n-gram boundary.
pub trait WeightFn: Send + Sync {
    fn weight(&self, a: u8, b: u8) -> u32;
}

// ---------------------------------------------------------------------------
// NGram
// ---------------------------------------------------------------------------

/// A single sparse n-gram token extracted from a byte slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NGram {
    /// The raw bytes of this n-gram.
    pub bytes: Vec<u8>,
    /// Byte offset of the first byte in the source.
    pub offset: usize,
}

impl NGram {
    pub fn new(bytes: Vec<u8>, offset: usize) -> Self {
        Self { bytes, offset }
    }

    /// FNV-1a 64-bit hash of the n-gram bytes, used as the lookup key.
    pub fn hash(&self) -> u64 {
        fnv_hash(&self.bytes)
    }
}

impl Hash for NGram {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bytes.hash(state);
    }
}

/// FNV-1a 64-bit hash of a byte slice.
pub fn fnv_hash(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 14695981039346656037;
    const PRIME: u64 = 1099511628211;
    let mut h = OFFSET_BASIS;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
const FNV_PRIME: u64 = 1099511628211;

// ---------------------------------------------------------------------------
// build_all
// ---------------------------------------------------------------------------

/// Extract **all** sparse n-grams from `input`.
///
/// Algorithm:
/// 1. Compute a weight for every adjacent byte pair.
/// 2. A token starts at index `i` if `weight[i-1] >= weight[i]` (or `i == 0`).
/// 3. A token ends   at index `i` if `weight[i] >= weight[i+1]` (or `i` is the last pair).
/// 4. We emit every maximal sub-sequence `[start, end]` of consecutive pairs
///    whose leftmost and rightmost pair weights are both strictly greater than
///    all interior pair weights.
///
/// Minimum token length: 2 bytes (one pair).
pub fn build_all(input: &[u8], weight_fn: &dyn WeightFn) -> Vec<NGram> {
    let mut ngrams = Vec::new();
    for_each_ngram(input, weight_fn, |start, end, _hash| {
        ngrams.push(NGram::new(input[start..end].to_vec(), start));
    });
    ngrams
}

/// Extract the FNV-1a hashes of every sparse n-gram in `input`.
///
/// This is the indexing fast path: it produces exactly the hashes of the
/// n-grams [`build_all`] would emit (in the same order) but allocates no
/// per-n-gram byte vectors and computes each hash incrementally in O(1).
pub fn build_all_hashes(input: &[u8], weight_fn: &dyn WeightFn) -> Vec<u64> {
    // Empirically ~2–3 n-grams are emitted per input byte; pre-allocate to
    // avoid repeated reallocation on large files.
    let mut hashes = Vec::with_capacity(input.len().saturating_mul(3));
    for_each_ngram(input, weight_fn, |_start, _end, hash| hashes.push(hash));
    hashes
}

/// Core extraction loop shared by [`build_all`] and [`build_all_hashes`].
///
/// Calls `emit(start_offset, end_offset_exclusive, fnv_hash)` for every valid
/// n-gram, in ascending `(start, end)` order.
///
/// We emit n-grams by finding all valid [start_pair .. end_pair] windows.
/// A window [s, e] is valid if:
///   weights[s] > weights[k]  for all s < k < e
///   weights[e] > weights[k]  for all s < k < e
/// (Both boundary weights strictly dominate all interior weights.)
///
/// # Performance
///
/// For a fixed start `s`, the running interior maximum `max_interior` is
/// non-decreasing as the end `e` advances, so once
/// `max_interior >= weights[s]` no later `e` can be valid and the inner loop
/// stops early.  This makes extraction near-linear in practice (proportional
/// to the number of emitted n-grams) instead of quadratic in the input size.
/// The FNV-1a hash of each n-gram is maintained incrementally while the
/// window extends, so no per-n-gram allocation or re-hashing is needed.
pub fn for_each_ngram<W: WeightFn + ?Sized>(
    input: &[u8],
    weight_fn: &W,
    mut emit: impl FnMut(usize, usize, u64),
) {
    if input.len() < 2 {
        return;
    }

    let n = input.len() - 1; // number of adjacent pairs
    let weights: Vec<u32> = (0..n)
        .map(|i| weight_fn.weight(input[i], input[i + 1]))
        .collect();

    for s in 0..n {
        let ws = weights[s];

        // FNV-1a over the first two bytes of any token starting at `s`.
        let mut hash = FNV_OFFSET_BASIS;
        hash = (hash ^ input[s] as u64).wrapping_mul(FNV_PRIME);
        hash = (hash ^ input[s + 1] as u64).wrapping_mul(FNV_PRIME);

        // Single-pair n-gram is always valid (no interior).
        emit(s, s + 2, hash);

        let mut max_interior: u32 = 0;
        let mut e = s + 1;
        while e < n {
            // Interior pairs are [s+1 .. e-1]; when extending past e-1 the
            // interior grows to include pair e-1 (only when e >= s+2).
            if e >= s + 2 {
                max_interior = max_interior.max(weights[e - 1]);
            }
            // `max_interior` never decreases, so once it reaches the left
            // boundary weight, `weights[s] > max_interior` can never hold
            // again for this `s` — stop scanning.
            if ws <= max_interior {
                break;
            }
            // Extend the token to end at pair `e`: append byte input[e + 1].
            hash = (hash ^ input[e + 1] as u64).wrapping_mul(FNV_PRIME);
            if weights[e] > max_interior {
                emit(s, e + 2, hash);
            }
            e += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// build_covering
// ---------------------------------------------------------------------------

/// Extract the **minimum covering** set of sparse n-grams from `input`.
///
/// "Covering" means: the union of byte ranges of the emitted n-grams spans
/// every byte in `input` (except possibly the first and last byte which cannot
/// form a pair boundary on both sides).
///
/// Strategy: greedy scan from left to right.  At each position, emit the
/// longest valid n-gram starting there, then advance past it.
pub fn build_covering(input: &[u8], weight_fn: &dyn WeightFn) -> Vec<NGram> {
    if input.len() < 2 {
        return vec![];
    }

    let n = input.len() - 1;
    let weights: Vec<u32> = (0..n)
        .map(|i| weight_fn.weight(input[i], input[i + 1]))
        .collect();

    let mut ngrams = Vec::new();
    let mut pos = 0usize; // current pair index

    while pos < n {
        // Find the longest valid n-gram starting at `pos`.
        let best = longest_ngram_from(pos, &weights, input);
        let end_pair = best; // inclusive pair index
        ngrams.push(NGram::new(input[pos..=end_pair + 1].to_vec(), pos));
        // Advance: next n-gram can share the last byte (overlap by 1 byte = 0 pairs)
        pos = end_pair + 1;
    }

    ngrams
}

/// Returns the pair index of the *last* pair of the longest valid n-gram
/// starting at pair `start`.
fn longest_ngram_from(start: usize, weights: &[u32], _input: &[u8]) -> usize {
    let n = weights.len();
    let mut max_interior: u32 = 0;
    let mut best = start; // at minimum the single-pair n-gram

    for e in (start + 1)..n {
        // Interior is [start+1 .. e-1]; when extending to e, interior grows to include e-1.
        // When e == start+1, there is no interior (empty range).
        // Only update max_interior when e >= start+2.
        if e >= start + 2 {
            max_interior = max_interior.max(weights[e - 1]);
        }
        if weights[start] > max_interior && weights[e] > max_interior {
            best = e;
        } else if weights[e] <= max_interior {
            // Further extension cannot improve: right boundary weight too low.
            // But we might still find a valid end later, so continue.
        }
    }

    best
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frequency_table::default_weight_fn;

    #[test]
    fn test_build_all_short() {
        let wf = default_weight_fn();
        // Single byte — no pairs.
        assert!(build_all(b"a", wf.as_ref()).is_empty());
        // Two bytes — one pair, one n-gram.
        let ng = build_all(b"ab", wf.as_ref());
        assert_eq!(ng.len(), 1);
        assert_eq!(ng[0].bytes, b"ab");
    }

    #[test]
    fn test_build_all_contains_single_pairs() {
        let wf = default_weight_fn();
        let input = b"hello";
        let all = build_all(input, wf.as_ref());
        // Every adjacent pair must appear as a single-pair n-gram.
        for i in 0..input.len() - 1 {
            let pair: Vec<u8> = input[i..=i + 1].to_vec();
            assert!(
                all.iter().any(|ng| ng.bytes == pair),
                "missing pair {:?} at {}",
                pair,
                i
            );
        }
    }

    #[test]
    fn test_covering_is_subset_of_all() {
        let wf = default_weight_fn();
        let input = b"fn main() { println!(\"hello world\"); }";
        let all = build_all(input, wf.as_ref());
        let covering = build_covering(input, wf.as_ref());

        let all_set: std::collections::HashSet<Vec<u8>> =
            all.iter().map(|ng| ng.bytes.clone()).collect();

        for ng in &covering {
            assert!(
                all_set.contains(&ng.bytes),
                "covering n-gram {:?} not in build_all result",
                String::from_utf8_lossy(&ng.bytes)
            );
        }
    }

    #[test]
    fn test_covering_spans_all_bytes() {
        let wf = default_weight_fn();
        let input = b"fn main() -> i32 { 42 }";
        let covering = build_covering(input, wf.as_ref());

        // Build a coverage bitmap over pair indices.
        let n = input.len() - 1;
        let mut covered = vec![false; n];
        for ng in &covering {
            let start = ng.offset;
            let end = start + ng.bytes.len() - 2; // last pair index
            for covered_pair in covered.iter_mut().take(end + 1).skip(start) {
                *covered_pair = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "some pairs not covered");
    }

    #[test]
    fn test_ngram_hash_stability() {
        let ng1 = NGram::new(b"abc".to_vec(), 0);
        let ng2 = NGram::new(b"abc".to_vec(), 5);
        // Hash is content-based, not offset-based.
        assert_eq!(ng1.hash(), ng2.hash());

        let ng3 = NGram::new(b"abcd".to_vec(), 0);
        assert_ne!(ng1.hash(), ng3.hash());
    }

    #[test]
    fn test_uniform_weight_all_single_pairs() {
        // With uniform weights, 2-pair n-grams can be emitted (empty interior),
        // but 3+ pair n-grams cannot (interior weights equal to boundaries).
        struct Uniform;
        impl WeightFn for Uniform {
            fn weight(&self, _: u8, _: u8) -> u32 {
                1
            }
        }
        let input = b"abcde";
        let all = build_all(input, &Uniform);
        // 4 single-pair n-grams + 3 two-pair n-grams = 7 total.
        assert_eq!(all.len(), 7);
        // Check we get both single and 2-pair ngrams
        assert!(all.iter().any(|ng| ng.bytes.len() == 2));
        assert!(all.iter().any(|ng| ng.bytes.len() == 3));
        // But no 3+ pair ngrams
        assert!(all.iter().all(|ng| ng.bytes.len() <= 3));
    }

    #[test]
    fn test_build_all_emits_long_ngram_with_skewed_weights() {
        // Test for T1: With skewed weights [5,1,1,5], we should get
        // a long n-gram covering all 4 pairs.
        struct Skewed;
        impl WeightFn for Skewed {
            fn weight(&self, a: u8, b: u8) -> u32 {
                // Map to indices: assume input is sequential bytes starting at some value
                // For simplicity, use a pattern that gives us [5,1,1,5]
                // If a=0,b=1: weight 5; a=1,b=2: weight 1; a=2,b=3: weight 1; a=3,b=4: weight 5
                match (a, b) {
                    (0, 1) => 5,
                    (1, 2) => 1,
                    (2, 3) => 1,
                    (3, 4) => 5,
                    _ => 1,
                }
            }
        }
        let input = &[0u8, 1, 2, 3, 4];
        let all = build_all(input, &Skewed);
        // Should have at least one n-gram with length >= 5 bytes (4 pairs)
        let has_long = all.iter().any(|ng| ng.bytes.len() >= 5);
        assert!(
            has_long,
            "Expected at least one n-gram with >= 5 bytes, got max length: {}",
            all.iter().map(|ng| ng.bytes.len()).max().unwrap_or(0)
        );
    }

    // -----------------------------------------------------------------------
    // Differential tests: optimised loop vs. the original O(n²) reference
    // -----------------------------------------------------------------------

    /// The original quadratic implementation, kept verbatim as a reference
    /// oracle for differential testing of the optimised loop.
    fn build_all_reference(input: &[u8], weight_fn: &dyn WeightFn) -> Vec<NGram> {
        if input.len() < 2 {
            return vec![];
        }

        let n = input.len() - 1; // number of adjacent pairs
        let weights: Vec<u32> = (0..n)
            .map(|i| weight_fn.weight(input[i], input[i + 1]))
            .collect();

        let mut ngrams = Vec::new();

        for s in 0..n {
            let mut max_interior: u32 = 0;
            for e in s..n {
                if e == s {
                    ngrams.push(NGram::new(input[s..=s + 1].to_vec(), s));
                } else {
                    if e >= s + 2 {
                        max_interior = max_interior.max(weights[e - 1]);
                    }
                    if weights[s] > max_interior && weights[e] > max_interior {
                        ngrams.push(NGram::new(input[s..=e + 1].to_vec(), s));
                    }
                }
            }
        }

        ngrams
    }

    /// Simple deterministic PRNG (xorshift64*) so the test needs no extra deps.
    struct XorShift(u64);

    impl XorShift {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }
    }

    fn assert_same_ngrams(input: &[u8], wf: &dyn WeightFn) {
        let fast = build_all(input, wf);
        let reference = build_all_reference(input, wf);
        assert_eq!(
            fast.len(),
            reference.len(),
            "n-gram count mismatch for input {:?}",
            String::from_utf8_lossy(&input[..input.len().min(64)])
        );
        for (i, (a, b)) in fast.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
                (a.offset, &a.bytes),
                (b.offset, &b.bytes),
                "n-gram #{i} mismatch"
            );
        }

        // Hash fast path must agree with hashing the emitted n-gram bytes.
        let hashes = build_all_hashes(input, wf);
        assert_eq!(hashes.len(), reference.len(), "hash count mismatch");
        for (i, (h, ng)) in hashes.iter().zip(reference.iter()).enumerate() {
            assert_eq!(*h, fnv_hash(&ng.bytes), "hash #{i} mismatch");
            assert_eq!(*h, ng.hash(), "NGram::hash #{i} mismatch");
        }
    }

    #[test]
    fn test_fast_path_matches_reference_edge_cases() {
        let wf = default_weight_fn();
        for input in &[
            &b""[..],
            &b"a"[..],
            &b"ab"[..],
            &b"abc"[..],
            &b"aaaaaa"[..],
            &b"\x00\x00\x00\x00"[..],
            &b"fn main() { println!(\"hello\"); }"[..],
            &b"                x"[..], // long equal-weight plateau then a byte
            &b"package main\n\nimport \"fmt\"\n"[..],
        ] {
            assert_same_ngrams(input, wf.as_ref());
        }
    }

    #[test]
    fn test_fast_path_matches_reference_random() {
        let wf = default_weight_fn();
        let mut rng = XorShift(0x9E3779B97F4A7C15);
        for round in 0..200 {
            let len = (rng.next() % 300) as usize;
            let mut input = Vec::with_capacity(len);
            for _ in 0..len {
                // Mix of ASCII-ish and arbitrary bytes to vary weight patterns.
                input.push(match rng.next() % 4 {
                    0 => b'a' + (rng.next() % 26) as u8,
                    1 => b' ' + (rng.next() % 16) as u8,
                    2 => b'0' + (rng.next() % 10) as u8,
                    _ => rng.next() as u8,
                });
            }
            assert_same_ngrams(&input, wf.as_ref());
            let _ = round;
        }
    }

    #[test]
    fn test_fast_path_matches_reference_custom_weight_fn() {
        // A weight function that can return 0 exercises the boundary
        // conditions of the early-termination logic.
        struct Zeroy;
        impl WeightFn for Zeroy {
            fn weight(&self, a: u8, b: u8) -> u32 {
                ((a as u32).wrapping_mul(31).wrapping_add(b as u32)) % 5
            }
        }
        let mut rng = XorShift(0xDEADBEEFCAFEF00D);
        for _ in 0..100 {
            let len = (rng.next() % 128) as usize;
            let input: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            assert_same_ngrams(&input, &Zeroy);
        }
    }
}
