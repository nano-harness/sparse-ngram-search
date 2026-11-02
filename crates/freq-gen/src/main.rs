//! Frequency table generator for `sparse-ngram-search`.
//!
//! Scans a corpus of source-code files, counts byte-pair frequencies,
//! and writes a 256×256 inverse-frequency weight table to a binary file.
//!
//! # Usage
//!
//! ```text
//! freq-gen --input /path/to/corpus --output crates/core/data/frequency_table.bin
//! ```
//!
//! The output file is 131072 bytes (65536 × 2 bytes, little-endian u16).
//! Copy it to `crates/core/data/frequency_table.bin` and recompile core.

use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use clap::Parser;
use ignore::WalkBuilder;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "freq-gen",
    about = "Generate byte-pair frequency table for sngs"
)]
struct Args {
    /// Input directory containing source code files.
    #[arg(short, long)]
    input: PathBuf,

    /// Output file path for the frequency table (binary format).
    #[arg(short, long, default_value = "frequency_table.bin")]
    output: PathBuf,

    /// Minimum file size in bytes to process.
    #[arg(long, default_value = "16")]
    min_size: u64,

    /// Maximum file size in bytes to process (0 = unlimited).
    #[arg(long, default_value = "0")]
    max_size: u64,

    /// Respect .gitignore / ignore files.
    #[arg(long, default_value = "true")]
    use_ignore: bool,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse();

    println!("Scanning {:?} …", args.input);

    // Collect file paths.
    let paths: Vec<PathBuf> = WalkBuilder::new(&args.input)
        .hidden(false)
        .ignore(args.use_ignore)
        .git_ignore(args.use_ignore)
        .build()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|ft| ft.is_file()).unwrap_or(false))
        .filter(|e| {
            let size = e.metadata().map(|m| m.len()).unwrap_or(0);
            size >= args.min_size && (args.max_size == 0 || size <= args.max_size)
        })
        .filter(|e| !is_binary(e.path()))
        .map(|e| e.path().to_path_buf())
        .collect();

    println!("Found {} candidate files", paths.len());

    // Parallel count accumulation.
    let counts: Arc<Mutex<Box<[u64; 65536]>>> = Arc::new(Mutex::new(Box::new([0u64; 65536])));

    let file_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    paths.par_iter().for_each(|path| {
        match std::fs::read(path) {
            Err(_) => {}
            Ok(data) => {
                // Skip binary content.
                if data.contains(&0u8) {
                    return;
                }
                // Count pairs locally.
                let mut local = [0u64; 65536];
                for pair in data.windows(2) {
                    local[pair[0] as usize * 256 + pair[1] as usize] += 1;
                }
                // Merge into global counts.
                let mut global = counts.lock().unwrap();
                for i in 0..65536 {
                    global[i] = global[i].saturating_add(local[i]);
                }
                file_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    });

    let processed = file_count.load(std::sync::atomic::Ordering::Relaxed);
    println!("Processed {} files", processed);

    let counts = Arc::try_unwrap(counts).unwrap().into_inner().unwrap();

    // Convert counts → inverse-frequency weights.
    let weights = counts_to_weights(&counts);

    // Write output.
    write_weights(&weights, &args.output)?;
    println!("Written to {:?}", args.output);

    // Print some statistics.
    print_stats(&weights, &counts);

    Ok(())
}

// ---------------------------------------------------------------------------
// Count → weight conversion
// ---------------------------------------------------------------------------

fn counts_to_weights(counts: &[u64; 65536]) -> Box<[u16; 65536]> {
    let max_count = *counts.iter().max().unwrap_or(&0);
    if max_count == 0 {
        eprintln!("warning: empty corpus, using uniform weights");
        return Box::new([1u16; 65536]);
    }

    let max_weight: f64 = 65535.0;
    let log_max = ((max_count + 1) as f64).ln();

    let mut weights = Box::new([0u16; 65536]);
    for i in 0..65536 {
        let c = counts[i];
        let log_freq = ((c + 1) as f64).ln();
        let scaled = (log_freq / log_max) * (max_weight - 1.0);
        let w = (max_weight - scaled).max(1.0).min(max_weight) as u16;
        weights[i] = w;
    }
    weights
}

fn write_weights(weights: &[u16; 65536], path: &std::path::Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create dir {:?}", parent))?;
    }
    let mut f = std::fs::File::create(path).with_context(|| format!("create {:?}", path))?;
    let mut buf = std::io::BufWriter::new(&mut f);
    for &w in weights.iter() {
        buf.write_u16::<LittleEndian>(w)?;
    }
    buf.flush()?;
    Ok(())
}

fn print_stats(weights: &[u16; 65536], counts: &[u64; 65536]) {
    let min_w = *weights.iter().min().unwrap();
    let max_w = *weights.iter().max().unwrap();
    let non_zero = counts.iter().filter(|&&c| c > 0).count();

    println!("Weight range    : {} – {}", min_w, max_w);
    println!(
        "Unique pairs    : {} / 65536 ({:.1}%)",
        non_zero,
        non_zero as f64 / 655.36
    );

    // Top 10 rarest pairs (highest weight, excluding zero counts).
    let mut indexed: Vec<(u16, usize)> = weights.iter().enumerate().map(|(i, &w)| (w, i)).collect();
    indexed.sort_by(|a, b| b.0.cmp(&a.0));
    println!("Top 10 rarest pairs (highest weight):");
    for (w, idx) in indexed.iter().take(10) {
        let a = (idx / 256) as u8;
        let b = (idx % 256) as u8;
        println!("  {:?}{:?}  weight={}", a as char, b as char, w);
    }
}

fn is_binary(path: &std::path::Path) -> bool {
    let binary_exts = [
        "png", "jpg", "jpeg", "gif", "bmp", "ico", "svg", "pdf", "zip", "tar", "gz", "bz2", "xz",
        "7z", "rar", "exe", "dll", "so", "dylib", "a", "o", "wasm", "bin", "dat", "lock", "mp3",
        "mp4", "wav", "ogg", "ttf", "woff", "woff2", "eot", "pyc", "class",
    ];
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        binary_exts.contains(&ext.to_lowercase().as_str())
    } else {
        false
    }
}
