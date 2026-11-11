#!/usr/bin/env python3
"""Dynamic INT8 quantization of an exported Demucs ONNX model.

Dynamic quantization stores weights as INT8 (~4x smaller than fp32) and
quantizes activations on the fly at run time -- no calibration dataset needed,
which suits us since the net is all conv/matmul. This is the weight-size lever
for the RK3566 device: 168 MB fp32 weights -> ~42 MB INT8.

We exclude the DFT STFT/iSTFT kernels (the _osp_* conv/conv_transpose) from
quantization: they are exact DSP, not learned, and INT8 there would add audible
error for no size win worth it. (They are matched by node-name substring.)

ASCII-only output.
"""
import argparse
import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.quant_utils import QuantFormat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hdemucs_mmi")
    ap.add_argument("--seconds", type=int, default=1)
    ap.add_argument("--out", default="../models")
    args = ap.parse_args()

    tag = "%s_%ds" % (args.model, args.seconds)
    src = os.path.join(args.out, tag + ".onnx")
    dst = os.path.join(args.out, tag + "_int8.onnx")

    # Keep the DFT STFT/iSTFT kernels in fp32: they are exact DSP, and INT8 on
    # them wrecks the output (max err ~4.0). They enter the graph as Constant
    # nodes (persistent=False buffers), so their Conv/ConvTranspose weight input
    # is NOT a named initializer -- unlike every learned layer, whose weight is
    # e.g. "encoder.5.conv.weight". Exclude exactly those Constant-fed convs.
    m = onnx.load(src, load_external_data=False)
    init_names = {i.name for i in m.graph.initializer}
    exclude = []
    for node in m.graph.node:
        if node.op_type in ("Conv", "ConvTranspose") and len(node.input) > 1:
            if node.input[1] not in init_names and node.name:
                exclude.append(node.name)
    print("excluding %d DFT-kernel conv nodes from quantization: %s"
          % (len(exclude), exclude))

    quantize_dynamic(
        src, dst,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=exclude,
        per_channel=True,
    )

    s0 = os.path.getsize(src) / 1e6
    s1 = os.path.getsize(dst) / 1e6
    print("%s: %.0f MB fp32 -> %.0f MB int8 (%.1fx)" % (tag, s0, s1, s0 / s1))
    print("wrote %s" % dst)


if __name__ == "__main__":
    main()
