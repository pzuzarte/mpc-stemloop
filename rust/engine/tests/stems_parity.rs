//! End-to-end parity: the Rust engine's 4 stems vs the Python reference
//! (`src/gen_stems_ref.py`). Needs the net ONNX (STEM_MODEL) and a libonnxruntime
//! (ORT_DYLIB_PATH); when they are absent (e.g. a plain host `cargo test`) the
//! test skips instead of failing.
#![allow(clippy::cast_possible_truncation)]

use std::fs;
use std::path::PathBuf;

use mpc_stemloop_engine::Engine;

fn fixtures() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p
}

fn read_f32(name: &str) -> Vec<f32> {
    let mut p = fixtures();
    p.push(name);
    let bytes = fs::read(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn meta() -> (usize, usize, usize, usize) {
    let mut p = fixtures();
    p.push("meta.txt");
    let s = fs::read_to_string(&p).unwrap_or_else(|e| panic!("read meta: {e}"));
    let v: Vec<usize> = s.split_whitespace().filter_map(|x| x.parse().ok()).collect();
    (v[0], v[1], v[2], v[3]) // channels, frames, secs, out_len
}

/// Peak resident set size in MB from /proc (Linux).
fn peak_rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<f64>().ok())
        })
        .map_or(0.0, |kb| kb / 1024.0)
}

fn errs(got: &[f32], want: &[f32]) -> (f64, f64) {
    assert_eq!(got.len(), want.len(), "stem length mismatch");
    let mut max_abs = 0.0_f64;
    let mut sq = 0.0_f64;
    for (&g, &w) in got.iter().zip(want.iter()) {
        let d = f64::from(g) - f64::from(w);
        max_abs = max_abs.max(d.abs());
        sq += d * d;
    }
    (max_abs, (sq / got.len() as f64).sqrt())
}

#[test]
fn engine_stems_match_python() {
    let Ok(model) = std::env::var("STEM_MODEL") else {
        eprintln!("SKIP: STEM_MODEL not set (no net ONNX / libonnxruntime here)");
        return;
    };
    let (channels, frames, secs, _out_len) = meta();
    let input = read_f32("input.bin");

    let mut engine = Engine::new(std::path::Path::new(&model), secs)
        .expect("build engine (needs ORT_DYLIB_PATH to a libonnxruntime)");
    let stems = engine.separate(&input, channels, frames).expect("separate");
    assert_eq!(stems.len(), 4);

    let mut worst = 0.0_f64;
    for (s, stem) in stems.iter().enumerate() {
        let (max_abs, rms) = errs(&stem.audio, &read_f32(&format!("stem{s}.bin")));
        println!("stem{s}: frames={} max_abs={max_abs:.3e} rms={rms:.3e}", stem.frames);
        worst = worst.max(max_abs);
    }
    println!("worst stem max_abs = {worst:.3e}");
    assert!(worst < 1e-3, "engine stems diverge from the Python reference");
}

/// Chunked path smoke test: a loop longer than the window must come back the
/// right length, all-finite, with peak RAM bounded to ~one window (not the whole
/// loop). Point STEM_MODEL at the 4 s bucket for this one.
#[test]
fn engine_chunked_smoke() {
    let Ok(model) = std::env::var("STEM_MODEL") else {
        eprintln!("SKIP: STEM_MODEL not set");
        return;
    };
    let window_secs = 4;
    let frames = 6 * 44_100; // 6 s loop -> chunked through the 4 s window
    let input: Vec<f32> = (0..frames * 2)
        .map(|i| (i as f32 * 0.002).sin() * 0.2)
        .collect();

    let mut engine =
        Engine::new(std::path::Path::new(&model), window_secs).expect("build engine (4s)");
    let stems = engine.separate(&input, 2, frames).expect("separate (chunked)");

    assert_eq!(stems.len(), 4);
    for (s, st) in stems.iter().enumerate() {
        assert_eq!(st.frames, frames, "stem {s} frame count");
        assert_eq!(st.audio.len(), frames * 2, "stem {s} buffer length");
        assert!(st.audio.iter().all(|v| v.is_finite()), "stem {s} has NaN/inf");
    }
    let rss = peak_rss_mb();
    println!("chunked 6s: 4 stems x {frames} frames, all finite; peak RSS {rss:.0} MB");
    // One 4 s window is ~835 MB; a non-chunked 6 s pass would be ~1 GB. Allow slack.
    assert!(rss < 1100.0, "peak RSS {rss:.0} MB too high -- chunking not bounding memory");
}
