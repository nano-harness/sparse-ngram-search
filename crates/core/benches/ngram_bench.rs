use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use sparse_ngram_core::{build_all, build_covering, default_weight_fn};

fn bench_build_all(c: &mut Criterion) {
    let wf = default_weight_fn();
    let samples: &[(&str, &[u8])] = &[
        ("short(32B)", b"fn main() { println!(\"hi\"); }"),
        ("medium(256B)", &[b'a'; 256]),
        ("long(4096B)", &[b'x'; 4096]),
    ];

    let mut group = c.benchmark_group("build_all");
    for (name, input) in samples {
        group.bench_with_input(BenchmarkId::from_parameter(name), input, |b, input| {
            b.iter(|| build_all(black_box(input), wf.as_ref()))
        });
    }
    group.finish();
}

fn bench_build_covering(c: &mut Criterion) {
    let wf = default_weight_fn();
    let samples: &[(&str, &[u8])] = &[
        ("short(32B)", b"fn main() { println!(\"hi\"); }"),
        ("medium(256B)", &[b'a'; 256]),
        ("long(4096B)", &[b'x'; 4096]),
    ];

    let mut group = c.benchmark_group("build_covering");
    for (name, input) in samples {
        group.bench_with_input(BenchmarkId::from_parameter(name), input, |b, input| {
            b.iter(|| build_covering(black_box(input), wf.as_ref()))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_build_all, bench_build_covering);
criterion_main!(benches);
