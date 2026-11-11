#!/usr/bin/env python3
"""Measure ORT peak memory + accuracy for an explicit ONNX path, reusing the
input/ref that test_onnx_runtime.py --make-ref saved for a given model/seconds.
ASCII-only output.
"""
import argparse
import os
import resource
import sys
import time
import numpy as np
import onnxruntime as ort

SR = 44100


def peak_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--model", default="hdemucs_mmi")
    ap.add_argument("--seconds", type=int, default=1)
    ap.add_argument("--lean", action="store_true")
    args = ap.parse_args()

    tag = "%s_%ds" % (args.model, args.seconds)
    x = np.load(os.path.join("/tmp", tag + "_in.npy"))
    ref_path = os.path.join("/tmp", tag + "_ref.npy")

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    if args.lean:
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    t1 = time.time()
    out = sess.run(None, {name: x})[0]
    run_s = time.time() - t1
    sz = os.path.getsize(args.onnx) / 1e6
    print("%s: model %.0f MB, input %d" % (os.path.basename(args.onnx), sz, x.shape[-1]))
    print("  ort run  : %.2f s" % run_s)
    print("  ort peak : %.0f MB" % peak_mb())
    if os.path.exists(ref_path):
        ref = np.load(ref_path)
        err = float(np.abs(out - ref).max())
        rms = float(np.sqrt(np.mean((out - ref) ** 2)))
        print("  max err  : %.2e   rms %.2e   %s"
              % (err, rms, "PASS" if err < 5e-2 else "FAIL"))


if __name__ == "__main__":
    main()
