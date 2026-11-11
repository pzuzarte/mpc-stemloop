#!/usr/bin/env python3
"""Measure ORT peak RAM + accuracy of the net-only (host-STFT-split) model.
Runs the ONNX net on saved (mag, mix), compares (xspec, xt) to the torch net
refs, and reports peak RSS. ASCII-only output.
"""
import argparse
import os
import resource
import sys
import time
import numpy as np
import onnxruntime as ort


def peak_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=1)
    ap.add_argument("--lean", action="store_true")
    args = ap.parse_args()
    s = args.seconds
    tag = "hdemucs_mmi_%ds" % s
    mag = np.load("/tmp/split_mmi_%ds_mag.npy" % s)
    mix = np.load("/tmp/split_mmi_%ds_mix.npy" % s)
    rx = np.load("/tmp/split_mmi_%ds_rx.npy" % s)
    rxt = np.load("/tmp/split_mmi_%ds_rxt.npy" % s)
    onnx_path = "../models/%s_net.onnx" % tag

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    if args.lean:
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    feed = {names[0]: mag, names[1]: mix}
    t1 = time.time()
    xspec, xt = sess.run(None, feed)
    run_s = time.time() - t1
    sz = os.path.getsize(onnx_path) / 1e6
    ex = float(np.abs(xspec - rx).max())
    et = float(np.abs(xt - rxt).max())
    print("%s_net: model %.0f MB, loop %ds" % (tag, sz, s))
    print("  ort run  : %.2f s" % run_s)
    print("  ort peak : %.0f MB" % peak_mb())
    print("  xspec err: %.2e   xt err: %.2e   %s"
          % (ex, et, "PASS" if max(ex, et) < 5e-3 else "FAIL"))


if __name__ == "__main__":
    main()
