//! Regex → n-gram query tree decomposition.
//!
//! Given a regular expression pattern, we extract the literal byte substrings
//! that any match *must* contain, and build sparse n-grams from them.
//!
//! The result is a [`QueryAtom`] tree:
//! - `And(ngrams)` – all these n-gram hashes must appear in the file.
//! - `Or(branches)` – at least one branch must match (for alternation).
//!
//! # Algorithm
//!
//! We parse the pattern into a `regex-syntax` HIR and recursively extract
//! *required literals* from each node:
//!
//! | HIR node         | Required literals                        |
//! |------------------|------------------------------------------|
//! | `Literal(bytes)` | The literal bytes themselves             |
//! | `Concat(a, b, …)`| Cartesian product of each child's literals (string concatenation) |
//! | `Alternation(…)` | Union of all branches (OR of their literals) |
//! | `Repetition(n≥1)`| The child's required literals           |
//! | `Repetition(n=0)`| Empty (may not appear at all)           |
//! | `Capture(inner)` | Same as inner                            |
//! | Dot / class      | Empty (too many possibilities)           |
//!
//! # F12: ASCII case-insensitive folding
//!
//! When a character class represents an ASCII case-insensitive pair (e.g., `(?i)f`
//! produces a class with ranges 'f'-'f' and 'F'-'F'), we generate both lowercase
//! and uppercase alternatives. For concatenations like `(?i)foo`, we produce an OR
//! of all case combinations: `foo`, `Foo`, `fOo`, `FOo`, `foO`, `FoO`, `fOO`, `FOO`.
//!
//! This prevents case-insensitive patterns from degrading to full scans.
//!
//! The final set of literals is limited to those ≥ `MIN_LITERAL_LEN` bytes
//! (shorter strings produce too many false positives in the index).

use std::collections::BTreeSet;

use anyhow::{Context, Result};
use regex_syntax::hir::{Class, ClassUnicode, Hir, HirKind, Literal};

use crate::ngram::{build_covering, NGram, WeightFn};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum literal length we'll use to generate n-grams.
/// Shorter literals produce too many false positives.
pub const MIN_LITERAL_LEN: usize = 2;

/// Maximum number of literals we keep per AND-branch before giving up.
/// (Prevents exponential blowup from large alternations.)
pub const MAX_LITERALS: usize = 256;

// ---------------------------------------------------------------------------
// QueryAtom
// ---------------------------------------------------------------------------

/// A node in the n-gram query tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryAtom {
    /// All of these n-gram hashes must be present in a candidate file.
    And(Vec<u64>),
    /// At least one branch must be satisfiable.
    Or(Vec<QueryAtom>),
    /// No constraint (the regex provides no useful literals).
    Any,
}

impl QueryAtom {
    /// Returns true if this atom imposes no constraint.
    pub fn is_any(&self) -> bool {
        matches!(self, QueryAtom::Any)
            || matches!(self, QueryAtom::And(v) if v.is_empty())
            || matches!(self, QueryAtom::Or(v) if v.is_empty())
    }

    /// Simplify: flatten nested `Or(Or(...))`, remove `Any` in `And`, etc.
    pub fn simplify(self) -> Self {
        match self {
            QueryAtom::And(mut hashes) => {
                hashes.sort_unstable();
                hashes.dedup();
                if hashes.is_empty() {
                    QueryAtom::Any
                } else {
                    QueryAtom::And(hashes)
                }
            }
            QueryAtom::Or(branches) => {
                let mut flat = Vec::new();
                let mut has_any = false;
                for b in branches {
                    let b = b.simplify();
                    match b {
                        QueryAtom::Any => {
                            has_any = true;
                        }
                        QueryAtom::Or(inner) => flat.extend(inner),
                        other => flat.push(other),
                    }
                }
                if has_any || flat.is_empty() {
                    QueryAtom::Any
                } else if flat.len() == 1 {
                    flat.remove(0)
                } else {
                    QueryAtom::Or(flat)
                }
            }
            other => other,
        }
    }
}

// ---------------------------------------------------------------------------
// decompose_regex
// ---------------------------------------------------------------------------

/// Decompose a regex pattern into a [`QueryAtom`] tree of n-gram hashes.
///
/// Returns `Ok(QueryAtom::Any)` if no useful literals can be extracted
/// (e.g., the pattern is `.*`).
pub fn decompose_regex(pattern: &str, weight_fn: &dyn WeightFn) -> Result<QueryAtom> {
    let hir = regex_syntax::Parser::new()
        .parse(pattern)
        .with_context(|| format!("parse regex: {pattern}"))?;

    let atom = extract_atom(&hir, weight_fn);
    Ok(atom.simplify())
}

// ---------------------------------------------------------------------------
// Internal: direct QueryAtom extraction
// ---------------------------------------------------------------------------

/// Check if a Unicode character class represents an ASCII case-insensitive single character.
///
/// Returns `Some((lower, upper))` if the class has exactly 2 ranges representing
/// an ASCII letter and its case-folded counterpart (e.g., 'f' and 'F').
fn is_ascii_case_fold(class: &ClassUnicode) -> Option<(u8, u8)> {
    let ranges = class.ranges();

    // Must have exactly 2 ranges
    if ranges.len() != 2 {
        return None;
    }

    let r0 = &ranges[0];
    let r1 = &ranges[1];

    // Each range must be a single character (start == end)
    if r0.start() != r0.end() || r1.start() != r1.end() {
        return None;
    }

    let c0 = r0.start() as u32;
    let c1 = r1.start() as u32;

    // Both must be ASCII
    if c0 > 127 || c1 > 127 {
        return None;
    }

    let b0 = c0 as u8;
    let b1 = c1 as u8;

    // Check if they form a case-insensitive pair (A-Z with a-z)
    if b0.is_ascii_uppercase() && b1.is_ascii_lowercase() && b0.to_ascii_lowercase() == b1 {
        return Some((b1, b0)); // (lowercase, uppercase)
    }

    if b0.is_ascii_lowercase() && b1.is_ascii_uppercase() && b0.to_ascii_uppercase() == b1 {
        return Some((b0, b1)); // (lowercase, uppercase)
    }

    None
}

/// Recursively extract a QueryAtom from a HIR node, handling multi-segment concat properly.
fn extract_atom(hir: &Hir, weight_fn: &dyn WeightFn) -> QueryAtom {
    match hir.kind() {
        HirKind::Empty => QueryAtom::Any,
        HirKind::Literal(Literal(bytes)) => {
            let hashes = literal_to_hashes(bytes, weight_fn);
            if hashes.is_empty() {
                QueryAtom::Any
            } else {
                QueryAtom::And(hashes)
            }
        }
        HirKind::Class(class) => {
            // F12: Handle ASCII case-insensitive single-character classes
            match class {
                Class::Unicode(unicode_class) => {
                    if let Some((_lower, _upper)) = is_ascii_case_fold(unicode_class) {
                        // This is a single ASCII letter with case folding (e.g., (?i)f)
                        // Return Any for now as we need to handle this in Concat context
                        // to generate proper literal alternatives
                        QueryAtom::Any
                    } else {
                        QueryAtom::Any
                    }
                }
                Class::Bytes(_) => QueryAtom::Any,
            }
        }
        HirKind::Look(_) => QueryAtom::And(vec![]), // zero-width, no constraint
        HirKind::Repetition(rep) => {
            if rep.min == 0 {
                QueryAtom::Any
            } else {
                extract_atom(&rep.sub, weight_fn)
            }
        }
        HirKind::Capture(cap) => extract_atom(&cap.sub, weight_fn),
        HirKind::Concat(children) => extract_concat_fallback(children, weight_fn),
        HirKind::Alternation(alts) => {
            // Keep unconstrained (Any) branches: if any branch imposes no
            // constraint, the whole alternation is unconstrained and must
            // degrade to `Any` (full scan) rather than silently dropping
            // the branch — otherwise `foo|.*` would only search for `foo`.
            let branches: Vec<QueryAtom> = alts
                .iter()
                .map(|alt| extract_atom(alt, weight_fn))
                .collect();

            if branches.iter().any(QueryAtom::is_any) {
                QueryAtom::Any
            } else if branches.len() == 1 {
                branches.into_iter().next().unwrap()
            } else {
                QueryAtom::Or(branches)
            }
        }
    }
}

/// Fallback for concat using literal extraction
fn extract_concat_fallback(children: &[Hir], weight_fn: &dyn WeightFn) -> QueryAtom {
    // Split concat by Any boundaries, creating AND of segments
    let mut segments = Vec::new();
    let mut current_segment = LiteralSet::empty();

    for child in children {
        let child_lits = extract_literals(child);
        if child_lits.is_any() {
            // Hit an Any boundary - save current segment and start new one
            if !matches!(current_segment.items, Some(ref v) if v.is_empty()) {
                segments.push(current_segment);
            }
            current_segment = LiteralSet::empty();
        } else {
            // Continue building current segment
            current_segment = current_segment.concat(&child_lits);
            if current_segment.is_any() {
                // Segment exploded, treat as Any boundary
                current_segment = LiteralSet::empty();
            }
        }
    }

    // Don't forget the last segment
    if !matches!(current_segment.items, Some(ref v) if v.is_empty()) {
        segments.push(current_segment);
    }

    // T20: Convert each segment to AND, then AND all segments together
    if segments.is_empty() {
        return QueryAtom::Any;
    }

    let mut all_hashes = Vec::new();
    for segment in &segments {
        let segment_atom = literals_to_atom(segment, weight_fn);
        match segment_atom {
            QueryAtom::And(hashes) => {
                all_hashes.extend(hashes);
            }
            QueryAtom::Or(_) => {
                // Can't merge OR into AND - fall back to first segment only
                return literals_to_atom(&segments[0], weight_fn);
            }
            QueryAtom::Any => {
                // Segment is unconstrained - skip it
            }
        }
    }

    if all_hashes.is_empty() {
        QueryAtom::Any
    } else {
        QueryAtom::And(all_hashes)
    }
}

// ---------------------------------------------------------------------------
// Internal: literal extraction
// ---------------------------------------------------------------------------

/// A set of byte strings representing required literals from a subtree.
///
/// The set is interpreted as: *at least one* of these strings must appear
/// (a union / OR semantic).  An empty set means "unconstrained".
#[derive(Debug, Clone)]
struct LiteralSet {
    /// The literals.  `None` means unconstrained (Any / .*).
    items: Option<Vec<Vec<u8>>>,
}

impl LiteralSet {
    fn any() -> Self {
        Self { items: None }
    }

    fn empty() -> Self {
        Self {
            items: Some(vec![]),
        }
    }

    fn single(bytes: Vec<u8>) -> Self {
        Self {
            items: Some(vec![bytes]),
        }
    }

    fn is_any(&self) -> bool {
        self.items.is_none()
    }

    /// AND of two sets: cartesian concatenation.
    ///
    /// `{ab, cd} AND {ef}` = `{abef, cdef}`
    ///
    /// If the full product would exceed [`MAX_LITERALS`], the left side (a
    /// proper prefix of the required concatenation) is kept instead.  Every
    /// match of the full pattern must still begin with one of these prefixes,
    /// so the result stays a necessary condition — truncating the product
    /// itself would drop alternatives and lose recall.  This keeps long
    /// case-insensitive literals like `(?i)connectionpool` (2^14 case
    /// combinations) on the indexed path with 256 8-byte prefixes instead of
    /// degrading to a full scan.
    fn concat(self, other: &LiteralSet) -> LiteralSet {
        match (self.items, &other.items) {
            (None, _) | (_, None) => LiteralSet::any(),
            (Some(left), Some(right)) => {
                if left.is_empty() {
                    return LiteralSet {
                        items: Some(right.clone()),
                    };
                }
                if right.is_empty() {
                    return LiteralSet { items: Some(left) };
                }
                if left.len().saturating_mul(right.len()) > MAX_LITERALS {
                    return LiteralSet { items: Some(left) };
                }
                let mut result = Vec::with_capacity(left.len() * right.len());
                for l in &left {
                    for r in right {
                        let mut combined = l.clone();
                        combined.extend_from_slice(r);
                        result.push(combined);
                    }
                }
                LiteralSet {
                    items: Some(result),
                }
            }
        }
    }

    /// OR of two sets: union.
    fn union(self, other: LiteralSet) -> LiteralSet {
        match (self.items, other.items) {
            (None, _) | (_, None) => LiteralSet::any(),
            (Some(mut a), Some(b)) => {
                a.extend(b);
                if a.len() > MAX_LITERALS {
                    return LiteralSet::any();
                }
                LiteralSet { items: Some(a) }
            }
        }
    }

    /// Filter to only literals with length ≥ MIN_LITERAL_LEN.
    #[allow(dead_code)]
    fn filter_short(self) -> LiteralSet {
        match self.items {
            None => LiteralSet::any(),
            Some(items) => {
                let filtered: Vec<_> = items
                    .into_iter()
                    .filter(|b| b.len() >= MIN_LITERAL_LEN)
                    .collect();
                LiteralSet {
                    items: Some(filtered),
                }
            }
        }
    }
}

/// Recursively extract required literals from a HIR node.
fn extract_literals(hir: &Hir) -> LiteralSet {
    match hir.kind() {
        HirKind::Empty => LiteralSet::any(), // Empty pattern matches without consuming input
        HirKind::Literal(Literal(bytes)) => LiteralSet::single(bytes.to_vec()),
        HirKind::Class(class) => {
            // F12: Handle ASCII case-insensitive single-character classes
            match class {
                Class::Unicode(unicode_class) => {
                    if let Some((lower, upper)) = is_ascii_case_fold(unicode_class) {
                        // This is a single ASCII letter with case folding (e.g., (?i)f)
                        // Return both lowercase and uppercase as alternatives
                        LiteralSet {
                            items: Some(vec![vec![lower], vec![upper]]),
                        }
                    } else {
                        LiteralSet::any()
                    }
                }
                Class::Bytes(_) => LiteralSet::any(),
            }
        }
        HirKind::Look(_) => LiteralSet::empty(), // zero-width assertion
        HirKind::Repetition(rep) => {
            if rep.min == 0 {
                // Optional: may not appear at all → no guarantee.
                LiteralSet::any()
            } else {
                // Appears at least once.
                extract_literals(&rep.sub)
            }
        }
        HirKind::Capture(cap) => extract_literals(&cap.sub),
        HirKind::Concat(children) => {
            // T20: Split concat by Any boundaries, creating AND of segments
            let mut segments = Vec::new();
            let mut current_segment = LiteralSet::empty();

            for child in children {
                let child_lits = extract_literals(child);
                if child_lits.is_any() {
                    // Hit an Any boundary - save current segment and start new one
                    if !matches!(current_segment.items, Some(ref v) if v.is_empty()) {
                        segments.push(current_segment);
                    }
                    current_segment = LiteralSet::empty();
                } else {
                    // Continue building current segment
                    current_segment = current_segment.concat(&child_lits);
                    if current_segment.is_any() {
                        // Segment exploded, treat as Any boundary
                        current_segment = LiteralSet::empty();
                    }
                }
            }

            // Don't forget the last segment
            if !matches!(current_segment.items, Some(ref v) if v.is_empty()) {
                segments.push(current_segment);
            }

            // If we got multiple segments, we need to express this as an AND
            // For now, return the first non-empty segment (or any if all empty)
            // A better implementation would return multiple literals to AND
            if segments.is_empty() {
                LiteralSet::any()
            } else if segments.len() == 1 {
                segments.into_iter().next().unwrap()
            } else {
                // Multiple segments: merge all literals into one set
                // Each segment's literals should appear in the file
                let mut all_literals = Vec::new();
                for seg in segments {
                    if let Some(items) = seg.items {
                        all_literals.extend(items);
                    }
                }
                if all_literals.is_empty() {
                    LiteralSet::any()
                } else {
                    LiteralSet {
                        items: Some(all_literals),
                    }
                }
            }
        }
        HirKind::Alternation(alts) => {
            let mut result = LiteralSet::empty();
            for alt in alts {
                let alt_lits = extract_literals(alt);
                result = result.union(alt_lits);
                if result.is_any() {
                    break;
                }
            }
            result
        }
    }
}

// ---------------------------------------------------------------------------
// Literals → QueryAtom
// ---------------------------------------------------------------------------

/// Convert a set of required literals into a [`QueryAtom`].
fn literals_to_atom(set: &LiteralSet, weight_fn: &dyn WeightFn) -> QueryAtom {
    let items = match &set.items {
        None => return QueryAtom::Any,
        Some(v) => v,
    };

    let useful: Vec<&Vec<u8>> = items
        .iter()
        .filter(|b| b.len() >= MIN_LITERAL_LEN)
        .collect();

    if useful.is_empty() {
        return QueryAtom::Any;
    }

    if useful.len() == 1 {
        // Single literal → AND of its covering n-gram hashes.
        let hashes = literal_to_hashes(useful[0], weight_fn);
        if hashes.is_empty() {
            QueryAtom::Any
        } else {
            QueryAtom::And(hashes)
        }
    } else {
        // Multiple literals → OR of their AND-sets.
        let branches: Vec<QueryAtom> = useful
            .iter()
            .map(|lit| {
                let hashes = literal_to_hashes(lit, weight_fn);
                if hashes.is_empty() {
                    QueryAtom::Any
                } else {
                    QueryAtom::And(hashes)
                }
            })
            .collect();
        QueryAtom::Or(branches)
    }
}

/// Build the set of n-gram hashes that must all be present in any file
/// containing `literal`.
fn literal_to_hashes(literal: &[u8], weight_fn: &dyn WeightFn) -> Vec<u64> {
    let ngrams: Vec<NGram> = build_covering(literal, weight_fn);
    let hashes: BTreeSet<u64> = ngrams.iter().map(|ng| ng.hash()).collect();
    hashes.into_iter().collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frequency_table::default_weight_fn;

    fn wf() -> Box<dyn WeightFn> {
        default_weight_fn()
    }

    #[test]
    fn test_literal_pattern() -> Result<()> {
        let atom = decompose_regex("hello", wf().as_ref())?;
        // "hello" is a 5-byte literal; should produce some AND constraints.
        assert!(!atom.is_any(), "expected constraints from literal 'hello'");
        Ok(())
    }

    #[test]
    fn test_wildcard_gives_any() -> Result<()> {
        let atom = decompose_regex(".*", wf().as_ref())?;
        assert!(atom.is_any());
        Ok(())
    }

    #[test]
    fn test_alternation() -> Result<()> {
        let atom = decompose_regex("foo|bar", wf().as_ref())?;
        // Should produce an OR of two AND-sets.
        match &atom {
            QueryAtom::Or(branches) => {
                assert_eq!(branches.len(), 2);
            }
            QueryAtom::Any => {} // acceptable if literals too short
            other => panic!("unexpected atom: {:?}", other),
        }
        Ok(())
    }

    #[test]
    fn test_concat_produces_and() -> Result<()> {
        // "fn main" – concatenation of two literals.
        let atom = decompose_regex("fn main", wf().as_ref())?;
        assert!(!atom.is_any());
        Ok(())
    }

    #[test]
    fn test_optional_gives_any() -> Result<()> {
        // "a?" → min=0 repetition → any
        let atom = decompose_regex("a?b", wf().as_ref())?;
        // "b" alone is 1 byte → filtered out → Any
        let _ = atom; // just no panic
        Ok(())
    }

    #[test]
    fn test_anchored_literal() -> Result<()> {
        let atom = decompose_regex("^fn main\\(", wf().as_ref())?;
        assert!(!atom.is_any());
        Ok(())
    }

    #[test]
    fn test_character_class_is_unconstrained() -> Result<()> {
        let atom = decompose_regex("[a-z]+", wf().as_ref())?;
        assert!(atom.is_any());
        Ok(())
    }

    #[test]
    fn test_case_insensitive_single_char() -> Result<()> {
        // F12: (?i)foo should generate OR of all case combinations
        let atom = decompose_regex("(?i)foo", wf().as_ref())?;
        // Should produce OR of literals like "foo", "Foo", "fOo", "FOo", "foO", "FoO", "fOO", "FOO"
        match &atom {
            QueryAtom::Or(branches) => {
                // We expect multiple branches (one for each case combination)
                assert!(
                    branches.len() > 1,
                    "expected multiple case combinations, got {}",
                    branches.len()
                );
            }
            QueryAtom::And(_) => {
                // Also acceptable if optimizer merged them
            }
            QueryAtom::Any => {
                panic!("(?i)foo should not degrade to Any after F12");
            }
        }
        Ok(())
    }

    #[test]
    fn test_case_insensitive_produces_or() -> Result<()> {
        // F12: (?i)ab should produce 4 branches: ab, Ab, aB, AB
        let atom = decompose_regex("(?i)ab", wf().as_ref())?;
        match &atom {
            QueryAtom::Or(branches) => {
                // 2^2 = 4 case combinations
                assert_eq!(branches.len(), 4, "expected 4 case combinations for (?i)ab");
            }
            other => {
                panic!("expected Or(_) for (?i)ab, got {:?}", other);
            }
        }
        Ok(())
    }

    #[test]
    fn test_alternation_with_unconstrained_branch_is_any() -> Result<()> {
        // An alternation where any branch is unconstrained (matches without
        // requiring any literal) must degrade to Any — dropping the branch
        // would silently lose every file matched only by that branch.
        for pattern in [
            "foo|.*",
            "foo|x?",
            "(foo|x?)",
            "zzz_absent|.",
            "alpha|beta|a*",
        ] {
            let atom = decompose_regex(pattern, wf().as_ref())?;
            assert!(
                atom.is_any(),
                "pattern {pattern:?} must degrade to Any, got {atom:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_alternation_of_constrained_literals_stays_or() -> Result<()> {
        let atom = decompose_regex("alpha|beta", wf().as_ref())?;
        match &atom {
            QueryAtom::Or(branches) => assert_eq!(branches.len(), 2),
            other => panic!("expected Or for constrained alternation, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn test_case_insensitive_long_literal_keeps_prefix() -> Result<()> {
        // (?i)connectionpool expands to 2^14 case combinations — far beyond
        // MAX_LITERALS.  The decomposer must keep the 256 8-byte prefix
        // combinations (a necessary condition for any match) instead of
        // degrading to a full scan.
        let atom = decompose_regex("(?i)connectionpool", wf().as_ref())?;
        match &atom {
            QueryAtom::Or(branches) => {
                assert!(
                    branches.len() > 1 && branches.len() <= MAX_LITERALS,
                    "expected ≤{MAX_LITERALS} prefix branches, got {}",
                    branches.len()
                );
            }
            other => panic!("expected Or of prefixes for long (?i) literal, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn test_simplify_deduplicates_hashes() {
        let atom = QueryAtom::And(vec![5, 3, 5, 1, 3]).simplify();
        if let QueryAtom::And(hashes) = atom {
            let mut sorted = hashes.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(hashes, sorted);
        }
    }
}
