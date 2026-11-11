#!/usr/bin/env python3
"""Generate an inference reference for the Rust ort spike (Phase 2).

Synthesizes a deterministic 1 s stereo loop, builds the net inputs (mag, mix) via
host_dsp, runs the net-only ONNX in onnxruntime (CPU), and writes mag/mix/xspec/xt
as raw little-endian f32 plus a manifest. The Rust spike loads mag+mix, runs the
same ONNX through the `ort` crate, and compares xspec/xt to this reference.

Also prints the Python-ORT peak RSS and net wall time as the baseline.

ASCII-only.
"""
import json
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import host_dsp as H

SR = 44100
MODEL = os.path.join(HERE, "..", "models", "hdemucs_mmi_1s_net.onnx")
OUT = os.path.abspath(os.path.join(HERE, "..", "rust", "spike", "fixtures"))
SEED = 7


def peak_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def f32(a):
    return np.ascontiguousarray(a, dtype="<f4")


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # Deterministic 1 s stereo "loop": a few tones + light noise, length = 1 s.
    t = np.arange(SR) / SR
    base = (0.3 * np.sin(2 * np.pi * 110 * t)
            + 0.2 * np.sin(2 * np.pi * 440 * t)
            + 0.1 * np.sin(2 * np.pi * 1760 * t))
    noise = 0.02 * rng.standard_normal((2, SR))
    mix = f32(np.stack([base, base * 0.9], axis=0) + noise)[None]   # (1,2,44100)

    zr, zi = H.spec_np(mix.astype(np.float64))
    mag = f32(H.magnitude_np(zr.astype(np.float32), zi.astype(np.float32)))

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    t0 = time.time()
    xspec, xt = sess.run(None, {names[0]: mag, names[1]: f32(mix)})
    net_s = time.time() - t0

    manifest = {}

    def dump(name, arr):
        a = f32(arr)
        a.tofile(os.path.join(OUT, name + ".bin"))
        manifest[name] = {"shape": list(a.shape), "dtype": "f32"}

    dump("mag", mag)
    dump("mix", mix)
    dump("xspec", xspec)
    dump("xt", xt)
    manifest["_meta"] = {
        "model": "hdemucs_mmi_1s_net.onnx",
        "input_names": names,
        "output_order": ["xspec", "xt"],
        "py_ort_net_s": round(net_s, 3),
        "py_ort_peak_mb": round(peak_mb(), 1),
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print("wrote reference to", OUT)
    for k, v in manifest.items():
        if not k.startswith("_"):
            print("  %-8s %s" % (k, v["shape"]))
    print("python ORT net wall: %.3f s" % net_s)
    print("python ORT peak RSS: %.1f MB" % peak_mb())


if __name__ == "__main__":
    main()
