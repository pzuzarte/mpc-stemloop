#!/usr/bin/env python3
"""Reference stems for the Rust engine parity test (Phase 4).

Synthesizes a deterministic sub-1s stereo loop (to exercise zero-padding + trim),
runs the lean pipeline exactly like src/stemloop.py (host_dsp + onnxruntime), and
writes the input and the four stems as raw INTERLEAVED little-endian f32 -- the
same layout the Rust engine produces. The Rust test reads input.bin, separates,
and compares to stem{0..3}.bin.

ASCII-only.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import host_dsp as H

SR = 44100
SECS = 1
FRAMES = 40000          # < SR: tests padding to the bucket and trim back
MODEL = os.path.join(HERE, "..", "models", "hdemucs_mmi_1s_net.onnx")
OUT = os.path.abspath(os.path.join(HERE, "..", "rust", "engine", "tests", "fixtures"))
SEED = 11


def f32(a):
    return np.ascontiguousarray(a, dtype="<f4")


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t = np.arange(FRAMES) / SR
    base = (0.3 * np.sin(2 * np.pi * 110 * t)
            + 0.2 * np.sin(2 * np.pi * 440 * t)
            + 0.1 * np.sin(2 * np.pi * 1760 * t))
    synth = np.stack([base, base * 0.9], axis=0) + 0.02 * rng.standard_normal((2, FRAMES))
    synth = f32(synth)                                   # (2, FRAMES) planar

    # input.bin: interleaved (FRAMES*2)
    f32(synth.T.reshape(-1)).tofile(os.path.join(OUT, "input.bin"))

    # lean pipeline (mirror stemloop.py)
    tgt = SECS * SR
    out_len = min(FRAMES, tgt)
    mix = synth[None]                                    # (1,2,FRAMES)
    mix_p = np.pad(mix, [(0, 0), (0, 0), (0, tgt - FRAMES)]).astype(np.float64)

    import onnxruntime as ort
    zr, zi = H.spec_np(mix_p)
    mag = H.magnitude_np(zr.astype(np.float32), zi.astype(np.float32))
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    nm = [i.name for i in sess.get_inputs()]
    xspec, xt = sess.run(None, {nm[0]: f32(mag), nm[1]: f32(mix_p)})
    stems = H.mask_ispec_np(xspec, out_len)[0] + xt[0][:, :, :out_len]   # (4,2,out_len)

    for s in range(4):
        f32(stems[s].T.reshape(-1)).tofile(os.path.join(OUT, "stem%d.bin" % s))

    with open(os.path.join(OUT, "meta.txt"), "w") as fh:
        fh.write("%d %d %d %d\n" % (2, FRAMES, SECS, out_len))   # channels frames secs out_len

    print("wrote stems reference to", OUT)
    print("  input  : %d frames x 2 ch (interleaved)" % FRAMES)
    for s in range(4):
        print("  stem%d  : peak %.4f" % (s, float(np.abs(stems[s]).max())))


if __name__ == "__main__":
    main()
