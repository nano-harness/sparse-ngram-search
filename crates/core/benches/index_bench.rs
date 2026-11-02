use std::path::PathBuf;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sparse_ngram_core::default_weight_fn;
use sparse_ngram_core::index::{build_postings_parallel, IndexBuilder};
use tempfile::TempDir;

fn make_files(n: usize, size: usize) -> Vec<(PathBuf, Vec<u8>)> {
    (0..n)
        .map(|i| {
            let name = PathBuf::from(format!("file{i}.rs"));
            // Pseudo-code content with varied byte patterns.
            let mut content = format!("// file {i}\nfn func_{i}() {{\n").into_bytes();
            for j in 0..size / 20 {
                content.extend_from_slice(format!("    let x{j} = {j} + {i};\n").as_bytes());
            }
            content.extend_from_slice(b"}\n");
            (name, content)
        })
        .collect()
}

fn bench_index_build_sequential(c: &mut Criterion) {
    let files = make_files(100, 1024);
    let tmp = TempDir::new().unwrap();

    c.bench_function("index_build_sequential_100files", |b| {
        b.iter(|| {
            let mut builder = IndexBuilder::with_default_weights();
            for (path, content) in &files {
                builder
                    .add_file(black_box(path), black_box(content))
                    .unwrap();
            }
            let _index = builder.build(tmp.path()).unwrap();
        })
    });
}

fn bench_index_build_parallel(c: &mut Criterion) {
    let files = make_files(100, 1024);
    let wf = default_weight_fn();

    c.bench_function("index_build_parallel_100files", |b| {
        b.iter(|| {
            let _postings = build_postings_parallel(black_box(&files), wf.as_ref());
        })
    });
}

criterion_group!(
    benches,
    bench_index_build_sequential,
    bench_index_build_parallel
);
criterion_main!(benches);
