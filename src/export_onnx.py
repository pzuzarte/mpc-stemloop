#!/usr/bin/env python3
"""Phase 3 spike: try to export a Demucs model to ONNX.

Why: under PyTorch, separating an 8 s loop peaks at ~1.8 GB, almost all of
it transient framework working memory (the model + runtime at rest is only
~450 MB). A lean C++ runtime (ONNX Runtime now, OpenVINO on the device)
should cut that dramatically. First we need the model in ONNX.

Demucs is known to resist ONNX export because of STFT / iSTFT and complex
number ops. This script attempts it at a FIXED input length (dynamic shapes
make it worse) and reports precisely where it breaks if it does.

ASCII-only output.
"""
import argparse
import os
import sys
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="htdemucs")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--opset", type=int, default=17,
                    help="opset 17+ is needed for the STFT op")
    ap.add_argument("--out", default="../models")
    ap.add_argument("--onnx-spec", action="store_true",
                    help="patch htdemucs to a complex-free STFT/iSTFT (ONNX-safe)")
    args = ap.parse_args()

    import torch
    from demucs.pretrained import get_model

    # htdemucs' transformer uses the fused MHA kernel (aten::_native_multi_head_
    # attention), which the ONNX exporter can't emit. Disable the fast path so
    # it decomposes into exportable ops.
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        print("MHA fastpath disabled")
    except Exception as e:
        print("could not disable MHA fastpath: %s" % e)

    os.makedirs(args.out, exist_ok=True)

    bag = get_model(args.model)
    bag.eval()
    # get_model returns a BagOfModels (does weighted averaging over >=1 nets).
    # Export the inner net itself, which has a plain forward.
    net = bag.models[0]
    net.eval()
    if args.onnx_spec:
        import htdemucs_onnx
        htdemucs_onnx.patch(net)
        print("applied onnx_spec patch (complex-free STFT/iSTFT)")
    sr = bag.samplerate
    print("model %s: inner net %s, sr=%d, sources=%s"
          % (args.model, type(net).__name__, sr, bag.sources))

    # A length the model considers valid. htdemucs is a fixed-length
    # transformer: valid_length() raises if asked for more than its training
    # segment, so clamp down and retry.
    want = int(args.seconds * sr)
    length = want
    if hasattr(net, "valid_length"):
        for cand in (want, int(getattr(net, "segment", 7.8) * sr), want):
            try:
                length = int(net.valid_length(cand))
                break
            except ValueError as e:
                print("valid_length rejected %d (%s), trying smaller" % (cand, e))
                length = min(cand, int(getattr(net, "segment", 7.8) * sr))
    print("using fixed input length: %d samples (%.2f s)" % (length, length / sr))

    dummy = torch.zeros(1, 2, length)  # (batch, stereo, samples)

    # Sanity: does the eager model even run at this shape?
    try:
        with torch.no_grad():
            y = net(dummy)
        print("eager forward OK, output shape %s" % (tuple(y.shape),))
    except Exception:
        print("eager forward FAILED:")
        traceback.print_exc()
        return 2

    out_path = os.path.join(args.out, "%s_%ds.onnx" % (args.model, int(args.seconds)))
    print("exporting to %s (opset %d) ..." % (out_path, args.opset))
    try:
        with torch.no_grad():
            torch.onnx.export(
                net, dummy, out_path,
                input_names=["mix"], output_names=["stems"],
                opset_version=args.opset, dynamo=False)
        print("EXPORT OK: %s (%.1f MB)"
              % (out_path, os.path.getsize(out_path) / 1e6))
        return 0
    except Exception:
        print("EXPORT FAILED (classic exporter):")
        traceback.print_exc()
        print("---- this is the expected STFT/complex sticking point ----")
        return 1


if __name__ == "__main__":
    sys.exit(main())
