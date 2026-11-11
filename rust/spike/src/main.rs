//! Phase 2 inference spike.
//!
//! Loads the net-only ONNX, feeds the `mag`/`mix` fixtures produced by
//! `src/gen_infer_ref.py`, runs inference through the `ort` crate (ONNX Runtime),
//! and compares `xspec`/`xt` to the Python-ORT reference. Prints op-coverage
//! success (it ran at all), parity error, net wall time and peak RSS.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

fn fixtures_dir() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("fixtures");
    p
}

fn read_f32(name: &str) -> Vec<f32> {
    let mut p = fixtures_dir();
    p.push(format!("{name}.bin"));
    let bytes = fs::read(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn model_path() -> PathBuf {
    if let Ok(p) = std::env::var("STEM_MODEL") {
        return PathBuf::from(p);
    }
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("../../models/hdemucs_mmi_1s_net.onnx");
    p
}

/// Peak resident set size in MB (ru_maxrss: bytes on macOS, KB on Linux).
fn peak_rss_mb() -> f64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    let r = usage.ru_maxrss as f64;
    if cfg!(target_os = "macos") {
        r / (1024.0 * 1024.0)
    } else {
        r / 1024.0
    }
}

fn errs(got: &[f32], want: &[f32]) -> (f64, f64) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for (&g, &w) in got.iter().zip(want.iter()) {
        let a = (f64::from(g) - f64::from(w)).abs();
        max_abs = max_abs.max(a);
        let r = a / (f64::from(w).abs() + 1e-6);
        max_rel = max_rel.max(r);
    }
    (max_abs, max_rel)
}

fn main() -> ort::Result<()> {
    let ones = std::env::args().any(|a| a == "ones");
    let (mag, mix) = if ones {
        println!("MODE: all-ones inputs");
        (
            vec![1.0_f32; 1 * 4 * 2048 * 44],
            vec![1.0_f32; 1 * 2 * 44100],
        )
    } else {
        (read_f32("mag"), read_f32("mix"))
    };
    let ref_xspec = read_f32("xspec");
    let ref_xt = read_f32("xt");

    let dsum = |v: &[f32]| v.iter().map(|&x| f64::from(x)).sum::<f64>();
    println!(
        "mag: len={} sum={:.6} [0..3]={:?}",
        mag.len(),
        dsum(&mag),
        &mag[..3]
    );
    println!(
        "mix: len={} sum={:.6} [0..3]={:?}",
        mix.len(),
        dsum(&mix),
        &mix[..3]
    );

    // Dump the exact bytes we are about to hand ORT, to diff against the fixtures.
    {
        let w = |name: &str, data: &[f32]| {
            let mut p = fixtures_dir();
            p.push(format!("rust_in_{name}.bin"));
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            fs::write(p, bytes).unwrap();
        };
        w("mag", &mag);
        w("mix", &mix);
    }

    let path = model_path();
    println!("loading {}", path.display());
    let load0 = Instant::now();
    let env = ort::init().with_name("stemloop-spike").build()?;
    let minimal = std::env::args().any(|a| a == "minimal");
    // Default to the VERIFIED-CORRECT config. ORT's ConstantFolding (and, with it,
    // ConstantSharing / CSE) mis-optimizes this graph *in the ort-crate path* and
    // corrupts the output; disabling those three at Level3 keeps conv fusions and
    // gives bit-exact parity with Python ORT. STEM_OPT overrides for diagnosis.
    let level = match std::env::var("STEM_OPT").as_deref() {
        Ok("disable") => GraphOptimizationLevel::Disable,
        Ok("1") => GraphOptimizationLevel::Level1,
        Ok("2") => GraphOptimizationLevel::Level2,
        _ => GraphOptimizationLevel::Level3,
    };
    // Default: disable the three faulty optimizers unless STEM_DISABLE_OPT is set
    // (empty string = disable nothing, for reproducing the bug).
    let disable = std::env::var("STEM_DISABLE_OPT").unwrap_or_else(|_| {
        "ConstantFolding;ConstantSharing;CommonSubexpressionElimination".into()
    });
    let mut session = if minimal {
        println!("MODE: minimal builder (no opt/threads set)");
        Session::builder(&env)?.commit_from_file(&path)?
    } else {
        println!("MODE: opt level {level:?}, disabled optimizers: [{disable}]");
        let mut b = Session::builder(&env)?.with_optimization_level(level)?;
        if !disable.is_empty() {
            b = b.with_config_entry("optimization.disable_specified_optimizers", &disable)?;
        }
        b.with_intra_threads(4)?.commit_from_file(&path)?
    };
    println!("session built in {:.2}s", load0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let outputs = if std::env::args().any(|a| a == "ndarray") {
        println!("MODE: ndarray input path");
        let mag_a = ndarray::Array4::from_shape_vec((1, 4, 2048, 44), mag).unwrap();
        let mix_a = ndarray::Array3::from_shape_vec((1, 2, 44100), mix).unwrap();
        session.run(ort::inputs![
            "mag" => Tensor::from_array(mag_a)?,
            "mix" => Tensor::from_array(mix_a)?
        ])?
    } else {
        let mag_t = Tensor::from_array(([1_usize, 4, 2048, 44], mag))?;
        let mix_t = Tensor::from_array(([1_usize, 2, 44100], mix))?;
        session.run(ort::inputs!["mag" => mag_t, "mix" => mix_t])?
    };
    let net_s = t0.elapsed().as_secs_f64();

    // Probe mode: dump every output by name as rust_probe_<name>.bin.
    if let Ok(list) = std::env::var("STEM_PROBE") {
        for name in list.split(',') {
            let (sh, data) = outputs[name].try_extract_tensor::<f32>()?;
            let s: f64 = data.iter().map(|&x| f64::from(x)).sum();
            println!("probe {name}: shape {sh:?} sum={s:.6}");
            let safe = name.replace(['/', '.'], "_");
            let mut p = fixtures_dir();
            p.push(format!("rust_probe_{safe}.bin"));
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            fs::write(p, bytes).unwrap();
        }
        return Ok(());
    }

    let (xspec_shape, xspec) = outputs["xspec"].try_extract_tensor::<f32>()?;
    let (xt_shape, xt) = outputs["xt"].try_extract_tensor::<f32>()?;
    println!("xspec shape {xspec_shape:?}");
    println!("xt    shape {xt_shape:?}");

    let (axs, rxs) = errs(xspec, &ref_xspec);
    let (axt, rxt) = errs(xt, &ref_xt);

    // Dump the Rust outputs for a full statistical compare in python.
    let dump = |name: &str, data: &[f32]| {
        let mut p = fixtures_dir();
        p.push(format!("rust_{name}.bin"));
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(p, bytes).unwrap();
    };
    dump("xspec", xspec);
    dump("xt", xt);

    println!("--- parity vs python ORT ---");
    println!("xspec: max_abs={axs:.3e} max_rel={rxs:.3e}");
    println!("xt   : max_abs={axt:.3e} max_rel={rxt:.3e}");
    println!("--- performance (host arm64) ---");
    println!("net wall : {net_s:.3} s");
    println!("peak RSS : {:.1} MB", peak_rss_mb());

    let tol = 1e-3;
    if axs < tol && axt < tol {
        println!("PASS: ort crate matches the Python ORT reference (tol {tol:.0e}).");
        Ok(())
    } else {
        println!("FAIL: output diverges from the reference.");
        std::process::exit(1)
    }
}
