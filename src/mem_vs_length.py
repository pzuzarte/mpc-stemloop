#!/usr/bin/env python3
"""Measure peak memory vs loop length for a model, in PyTorch (one process per
run so ru_maxrss is a clean per-config peak). Tests the core thesis: a
variable-length model (hdemucs_mmi) should use less memory at shorter lengths,
while htdemucs (fixed 7.8s segment) should not. ASCII-only output.
"""
import resource
import sys
import time
import torch
from demucs.pretrained import get_model

SR = 44100


def peak_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def main():
    model = sys.argv[1]
    seconds = float(sys.argv[2])
    torch.manual_seed(0)
    torch.set_num_threads(4)
    bag = get_model(model); bag.eval()
    net = bag.models[0]; net.eval()
    want = int(seconds * SR)
    L = want
    if hasattr(net, "valid_length"):
        try:
            L = int(net.valid_length(want))
        except Exception:
            L = want
    x = torch.randn(1, 2, L)
    t = time.time()
    with torch.no_grad():
        net(x)
    dt = time.time() - t
    print("%-13s %4.1fs  len=%7d  peak=%5.0f MB  run=%5.2fs"
          % (model, seconds, L, peak_mb(), dt))


if __name__ == "__main__":
    main()
