#!/usr/bin/env python3
"""Confirm the complex-free patch matches stock htdemucs numerically."""
import torch
from demucs.pretrained import get_model
import htdemucs_onnx

torch.manual_seed(0)


def main():
    bag = get_model("htdemucs")
    bag.eval()
    net = bag.models[0]
    net.eval()
    sr = bag.samplerate
    length = int(7.0 * sr)
    if hasattr(net, "valid_length"):
        length = int(net.valid_length(length))
    x = torch.randn(1, 2, length)

    with torch.no_grad():
        y_ref = net(x)
    htdemucs_onnx.patch(net)
    with torch.no_grad():
        y_patched = net(x)

    err = (y_patched - y_ref).abs().max().item()
    rel = err / (y_ref.abs().max().item() + 1e-9)
    print("output shape %s" % (tuple(y_ref.shape),))
    print("max abs err  = %.2e" % err)
    print("max rel err  = %.2e" % rel)
    print("RESULT:", "PASS" if err < 5e-3 else "FAIL")


if __name__ == "__main__":
    main()
