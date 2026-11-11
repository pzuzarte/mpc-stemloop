#!/usr/bin/env python3
"""Convert an exported Demucs ONNX model to FP16 weights.

INT8 dynamic quantization is too lossy for Demucs (drums/vocals SNR goes
negative). FP16 halves the weights with far smaller accuracy loss and is well
supported by ORT and the RK3566 NPU. We keep the DFT STFT/iSTFT Constant-fed
convs in fp32 (op_block_list on Conv/ConvTranspose would be too broad, so we
pass the specific node names), and keep the first/last ops in fp32 for range
safety. ASCII-only output.
"""
import argparse
import os
import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16


def dft_conv_names(model_proto):
    init_names = {i.name for i in model_proto.graph.initializer}
    names = []
    for node in model_proto.graph.node:
        if node.op_type in ("Conv", "ConvTranspose") and len(node.input) > 1:
            if node.input[1] not in init_names and node.name:
                names.append(node.name)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hdemucs_mmi")
    ap.add_argument("--seconds", type=int, default=1)
    ap.add_argument("--out", default="../models")
    args = ap.parse_args()

    tag = "%s_%ds" % (args.model, args.seconds)
    src = os.path.join(args.out, tag + ".onnx")
    dst = os.path.join(args.out, tag + "_fp16.onnx")

    m = onnx.load(src)
    # Keep the DFT STFT/iSTFT convs in fp32: fp16 whole-graph gives NaN because
    # the nfft=4096 DFT sums overflow fp16's +/-65504 range. INT8 is far worse
    # on them. So block them. The converter then inserts cast nodes to bridge
    # fp16<->fp32; where a shared tensor (e.g. a Pad output) feeds two blocked
    # convs it emits byte-identical duplicate Cast nodes (same input AND output
    # name), which ORT rejects. They are exact duplicates, so drop the extras.
    keep = dft_conv_names(m)
    print("keeping %d DFT-kernel conv nodes in fp32" % len(keep))
    m16 = convert_float_to_float16(m, keep_io_types=True, node_block_list=keep)
    seen_out = set()
    kept = []
    removed = 0
    for node in m16.graph.node:
        sig = (node.op_type, tuple(node.input), tuple(node.output))
        if node.output and tuple(node.output) in seen_out:
            removed += 1          # duplicate producer of an already-made tensor
            continue
        seen_out.add(tuple(node.output))
        kept.append(node)
    if removed:
        del m16.graph.node[:]
        m16.graph.node.extend(kept)
        print("removed %d duplicate cast nodes" % removed)
    onnx.save(m16, dst)

    s0 = os.path.getsize(src) / 1e6
    s1 = os.path.getsize(dst) / 1e6
    print("%s: %.0f MB fp32 -> %.0f MB fp16 (%.1fx)" % (tag, s0, s1, s0 / s1))
    print("wrote %s" % dst)


if __name__ == "__main__":
    main()
