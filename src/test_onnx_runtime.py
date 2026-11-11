#!/usr/bin/env python3
"""Verify an exported ONNX Demucs model and measure ONNX Runtime peak memory.

Parametric over model + loop length so we can compare a fixed-segment model
(htdemucs) against a variable-length one (hdemucs_mmi) at short lengths.

  --make-ref : import torch, build a fixed random input at the model's valid
               length for --seconds, run the patched net, save input + ref.
  (default)  : import ONLY onnxruntime + numpy, run the ONNX model on that
               input, print peak RSS and max error vs the reference.

ASCII-only output.
"""
import argparse
import os
import resource
import sys
import time
import numpy as np

SR = 44100
HERE = os.path.dirname(os.path.abspath(__file__))


def peak_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def paths(model, seconds):
    tag = "%s_%ds" % (model, int(seconds))
    return (os.path.join("/tmp", tag + "_in.npy"),
            os.path.join("/tmp", tag + "_ref.npy"),
            os.path.join(HERE, "..", "models", tag + ".onnx"))


def make_ref(model, seconds):
    import torch
    from demucs.pretrained import get_model
    import htdemucs_onnx
    inp, ref, _ = paths(model, seconds)
    torch.manual_seed(0)
    bag = get_model(model); bag.eval()
    net = bag.models[0]; net.eval()
    length = int(seconds * SR)
    if hasattr(net, "valid_length"):
        length = int(net.valid_length(length))
    x = torch.randn(1, 2, length, dtype=torch.float32)
    htdemucs_onnx.patch(net)
    with torch.no_grad():
        y = net(x)
    np.save(inp, x.numpy()); np.save(ref, y.numpy())
    print("%s %ds: input len %d, ref %s" % (model, seconds, length, tuple(y.shape)))


def measure(model, seconds, lean):
    import onnxruntime as ort
    inp, ref, onnx_path = paths(model, seconds)
    x = np.load(inp)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    if lean:
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    t1 = time.time()
    out = sess.run(None, {name: x})[0]
    run_s = time.time() - t1
    sz = os.path.getsize(onnx_path) / 1e6
    print("%s %ds: model %.0f MB, input %s" % (model, seconds, sz, x.shape[-1]))
    print("  ort run  : %.2f s" % run_s)
    print("  ort peak : %.0f MB" % peak_mb())
    if os.path.exists(ref):
        err = float(np.abs(out - np.load(ref)).max())
        print("  max err  : %.2e  %s" % (err, "PASS" if err < 5e-3 else "FAIL"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="htdemucs")
    ap.add_argument("--seconds", type=float, default=7.0)
    ap.add_argument("--make-ref", action="store_true")
    ap.add_argument("--lean", action="store_true")
    args = ap.parse_args()
    if args.make_ref:
        make_ref(args.model, args.seconds)
    else:
        measure(args.model, args.seconds, args.lean)
