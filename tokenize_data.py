"""
Convert byte shards -> BPE token shards (uint16).

Processes one byte shard at a time; with --delete-after, removes each byte
shard once its token shard is written (byte shards are mirrored at
ShadeEngine/fineweb-edu-bytes-10BT, so they are recoverable).

Output format mirrors the byte shards: flat uint16 ids, every document
preceded by the <|doc|> token (id 0).
"""

import argparse
import glob
import json
import os
import time

import numpy as np
from tokenizers import Tokenizer

DOC_SEP_BYTE = 254
DOC_TOKEN = 0  # <|doc|>


def docs_in_shard(path):
    arr = np.memmap(path, dtype=np.uint8, mode="r")
    seps = np.where(arr == DOC_SEP_BYTE)[0]
    bounds = list(seps) + [len(arr)]
    for i in range(len(seps)):
        yield bytes(arr[bounds[i] + 1: bounds[i + 1]]).decode("utf-8", errors="replace")


def tokenize_shard(tok, in_path, out_path, batch_docs=4000):
    n_tokens = 0
    n_bytes = os.path.getsize(in_path)
    with open(out_path, "wb") as f:
        batch = []
        def flush():
            nonlocal n_tokens
            if not batch:
                return
            encs = tok.encode_batch(batch)   # parallel across docs (rust/rayon)
            for e in encs:
                ids = np.asarray([DOC_TOKEN] + e.ids, dtype=np.uint16)
                ids.tofile(f)
                n_tokens += len(ids)
            batch.clear()
        for doc in docs_in_shard(in_path):
            batch.append(doc)
            if len(batch) >= batch_docs:
                flush()
        flush()
    return n_tokens, n_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/fineweb_edu_bytes")
    ap.add_argument("--out-dir", default="data/fineweb_edu_bpe")
    ap.add_argument("--tokenizer", default="tokenizer_32k.json")
    ap.add_argument("--delete-after", action="store_true",
                    help="delete each TRAIN byte shard after tokenizing (val is kept)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = Tokenizer.from_file(args.tokenizer)

    total_tokens = total_bytes = 0
    t0 = time.time()
    paths = sorted(glob.glob(os.path.join(args.in_dir, "*.bin")))
    for p in paths:
        name = os.path.basename(p)
        out = os.path.join(args.out_dir, name)
        nt, nb = tokenize_shard(tok, p, out)
        total_tokens += nt
        total_bytes += nb
        if args.delete_after and name.startswith("train_"):
            os.remove(p)
        print(f"[{time.time()-t0:6.0f}s] {name}: {nt/1e6:.1f}M tokens "
              f"({nb/nt:.3f} B/tok){'  [byte shard deleted]' if args.delete_after and name.startswith('train_') else ''}",
              flush=True)

    meta = {
        "tokenizer": args.tokenizer,
        "vocab_size": 32768,
        "dtype": "uint16",
        "doc_token": DOC_TOKEN,
        "total_tokens": total_tokens,
        "source_bytes": total_bytes,
        "avg_bytes_per_token": total_bytes / total_tokens,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
