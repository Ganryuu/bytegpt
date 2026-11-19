"""
Prepare FineWeb-Edu (sample-10BT) as raw bytes for byte-level language modeling.

nanoGPT-style output: flat binary shards of uint8, directly memmap-able.
  data/fineweb_edu_bytes/
    train_00000.bin ... (500 MB each)
    val_00000.bin
    meta.json

Byte vocabulary:
  0..255   raw UTF-8 bytes of the text
  254      <|doc|> document separator (0xFE never occurs in valid UTF-8)
  255      reserved (0xFF also never occurs in valid UTF-8)

Every 1000th document goes to the validation split.

Usage:
  python prepare_fineweb_bytes.py                 # full run (~45 GB)
  python prepare_fineweb_bytes.py --max-gb 0.2    # smoke test
"""

import argparse
import json
import os
import time

import numpy as np
from datasets import load_dataset

DOC_SEP = 254
SHARD_BYTES = 500 * 1024 * 1024  # 500 MB per train shard
VAL_EVERY = 1000  # every Nth doc -> val


class ShardWriter:
    def __init__(self, out_dir: str, split: str):
        self.out_dir = out_dir
        self.split = split
        self.shard_idx = 0
        self.bytes_in_shard = 0
        self.total_bytes = 0
        self.n_docs = 0
        self.f = None
        self._open()

    def _path(self):
        return os.path.join(self.out_dir, f"{self.split}_{self.shard_idx:05d}.bin")

    def _open(self):
        self.f = open(self._path(), "wb", buffering=1024 * 1024)

    def write_doc(self, data: bytes):
        # separator first so every doc starts with a <|doc|> byte
        self.f.write(bytes([DOC_SEP]))
        self.f.write(data)
        n = len(data) + 1
        self.bytes_in_shard += n
        self.total_bytes += n
        self.n_docs += 1
        if self.split == "train" and self.bytes_in_shard >= SHARD_BYTES:
            self.f.close()
            self.shard_idx += 1
            self.bytes_in_shard = 0
            self._open()

    def close(self):
        if self.f:
            self.f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fineweb_edu_bytes")
    ap.add_argument("--max-gb", type=float, default=None,
                    help="stop after this many GB of text (for smoke tests)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    max_bytes = int(args.max_gb * 1e9) if args.max_gb else None

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)

    train = ShardWriter(args.out, "train")
    val = ShardWriter(args.out, "val")

    t0 = time.time()
    last_report = t0
    for i, doc in enumerate(ds):
        data = doc["text"].encode("utf-8")
        # 0xFE/0xFF can't occur in valid UTF-8; be paranoid anyway
        if DOC_SEP in data or 255 in data:
            data = data.replace(bytes([DOC_SEP]), b"").replace(b"\xff", b"")
        w = val if i % VAL_EVERY == 0 else train
        w.write_doc(data)

        total = train.total_bytes + val.total_bytes
        now = time.time()
        if now - last_report > 30:
            gb = total / 1e9
            rate = total / (now - t0) / 1e6
            print(f"[{now - t0:7.0f}s] {gb:6.2f} GB  {train.n_docs + val.n_docs:>9} docs  {rate:6.1f} MB/s", flush=True)
            last_report = now
        if max_bytes and total >= max_bytes:
            print(f"hit --max-gb limit at {total/1e9:.2f} GB, stopping")
            break

    train.close()
    val.close()

    meta = {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "dtype": "uint8",
        "doc_separator_byte": DOC_SEP,
        "reserved_bytes": [255],
        "val_every_nth_doc": VAL_EVERY,
        "train_bytes": train.total_bytes,
        "train_docs": train.n_docs,
        "train_shards": train.shard_idx + 1,
        "val_bytes": val.total_bytes,
        "val_docs": val.n_docs,
        "shard_size_bytes": SHARD_BYTES,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
