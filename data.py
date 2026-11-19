"""
Data loading for byte-level LM training.

Shards are flat uint8 files (see prepare_fineweb_bytes.py). We memmap them
(the OS pages data in lazily; nothing is loaded up front) and sample random
crops. A training example is block_size+1 consecutive bytes: x = [0:n],
y = [1:n+1] (next-byte prediction targets).

Random crops can start mid-document and even mid-UTF-8-character. That is
intentional -- the model must learn UTF-8 structure itself, and document
boundaries are visible via the 0xFE separator byte.
"""

import glob
import json
import os

import numpy as np
import torch


class ByteShards:
    """Also serves BPE token shards: dtype is read from the directory's
    meta.json (uint8 for raw bytes, uint16 for token ids). 'bytes' in
    variable names then means 'units'."""

    def __init__(self, data_dir: str, split: str):
        meta_path = os.path.join(data_dir, "meta.json")
        self.meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        dtype = np.dtype(self.meta.get("dtype", "uint8"))
        paths = sorted(glob.glob(os.path.join(data_dir, f"{split}_*.bin")))
        assert paths, f"no {split} shards found in {data_dir}"
        self.dtype = dtype
        self.shards = [np.memmap(p, dtype=dtype, mode="r") for p in paths]
        sizes = np.array([len(s) for s in self.shards], dtype=np.float64)
        self.total_bytes = int(sizes.sum())
        # sample shards proportionally to size so every byte is equally likely
        self.probs = sizes / sizes.sum()

    def sample_batch(self, batch_size: int, block_size: int, rng: np.random.Generator,
                     device: str):
        xs = np.empty((batch_size, block_size + 1), dtype=self.dtype)
        shard_ids = rng.choice(len(self.shards), size=batch_size, p=self.probs)
        for i, sid in enumerate(shard_ids):
            shard = self.shards[sid]
            off = rng.integers(0, len(shard) - block_size - 1)
            xs[i] = shard[off: off + block_size + 1]
        buf = torch.from_numpy(xs.astype(np.int64))
        x = buf[:, :-1].to(device, non_blocking=True)
        y = buf[:, 1:].to(device, non_blocking=True)
        return x, y

    def sequential_windows(self, block_size: int, max_windows: int):
        """Deterministic non-overlapping windows for evaluation."""
        count = 0
        for shard in self.shards:
            for off in range(0, len(shard) - block_size - 1, block_size):
                yield shard[off: off + block_size + 1]
                count += 1
                if count >= max_windows:
                    return
