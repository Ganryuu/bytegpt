"""
Show where a trained H-Net cuts text into chunks.

    .venv-hnet/bin/python chunk_viz.py --ckpt runs/hnet-44B/ckpt.pt

Prints (1) the achieved bytes-per-chunk on each frozen eval domain and
(2) sample texts with a '|' wherever the routing module starts a new chunk.
"""

import argparse
import glob
import os

import numpy as np
import torch

from evaluate import DOC_SEP, load_model

SAMPLES = [
    "The French Revolution began in 1789 and radically transformed the country's government.",
    "Photosynthesis is the process by which green plants convert sunlight into chemical energy.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
    "La Révolution française commence en 1789 et transforme profondément la société.",
    "Teh qiuck borwn fox jumsp ovre the lzay dog, unbelievably and irresponsibly.",
]


@torch.no_grad()
def boundaries(hnet, ids: torch.Tensor):
    mask = torch.ones_like(ids, dtype=torch.bool)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = hnet(ids, mask=mask)
    return out.bpred_output[0].boundary_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--windows", type=int, default=32)
    args = ap.parse_args()

    shim, cfg, _ = load_model(args.ckpt, "cuda")
    hnet = shim.hnet
    T = cfg.block_size

    print("\n=== bytes per chunk by domain ===")
    for path in sorted(glob.glob("eval_data/*.bin")):
        arr = np.memmap(path, dtype=np.uint8, mode="r")
        n_win = min(len(arr) // T, args.windows)
        fracs = []
        for i in range(0, n_win, 8):
            rows = [arr[j * T:(j + 1) * T] for j in range(i, min(i + 8, n_win))]
            ids = torch.from_numpy(np.stack(rows).astype(np.int64)).cuda()
            fracs.append(boundaries(hnet, ids).float().mean().item())
        print(f"{os.path.basename(path)[:-4]:14s} N = {1 / np.mean(fracs):.2f}")

    print("\n=== chunk boundaries ('|' = a new chunk starts here) ===")
    for text in SAMPLES:
        raw = bytes([DOC_SEP]) + text.encode("utf-8")
        ids = torch.tensor(list(raw), dtype=torch.long, device="cuda")[None]
        b = boundaries(hnet, ids)[0].tolist()
        out = bytearray()
        for byte, is_start in list(zip(raw, b))[1:]:
            if is_start:
                out += b"|"
            out.append(byte)
        n = (len(raw) - 1) / max(sum(b[1:]), 1)
        print(f"\n[N = {n:.2f}]\n{out.decode('utf-8', errors='replace')}")


if __name__ == "__main__":
    main()
