"""
Build small, frozen evaluation sets in eval_data/. Each is a few MB of
uint8 bytes in the same format as training shards (0xFE doc separators).

Domains:
  fineweb_val  - held-out split of the training distribution (in-distribution)
  wiki_en      - English Wikipedia (near-distribution)
  wiki_fr      - French Wikipedia (out-of-distribution language)
  code_py      - Python source code (out-of-distribution domain)

Also downloads hellaswag_val.jsonl for the multiple-choice eval.

Frozen once, reused for every checkpoint and every architecture -- evals are
only comparable if the data never changes.
"""

import json
import os
import urllib.request

import numpy as np
from datasets import load_dataset

OUT = "eval_data"
DOC_SEP = 254
TARGET_BYTES = 5 * 1024 * 1024  # 5 MB per domain


def write_domain(name: str, texts):
    path = os.path.join(OUT, f"{name}.bin")
    total = 0
    with open(path, "wb") as f:
        for t in texts:
            data = t.encode("utf-8").replace(b"\xfe", b"").replace(b"\xff", b"")
            f.write(bytes([DOC_SEP]) + data)
            total += len(data) + 1
            if total >= TARGET_BYTES:
                break
    print(f"{name}: {total/1e6:.1f} MB")


def main():
    os.makedirs(OUT, exist_ok=True)

    # in-distribution: reuse our existing val split
    val = np.memmap("data/fineweb_edu_bytes/val_00000.bin", dtype=np.uint8, mode="r")
    val[:TARGET_BYTES].tofile(os.path.join(OUT, "fineweb_val.bin"))
    print(f"fineweb_val: {TARGET_BYTES/1e6:.1f} MB")

    wiki_en = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    write_domain("wiki_en", (d["text"] for d in wiki_en))

    wiki_fr = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)
    write_domain("wiki_fr", (d["text"] for d in wiki_fr))

    code = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)
    write_domain("code_py", (d["content"] for d in code))

    hs_path = os.path.join(OUT, "hellaswag_val.jsonl")
    if not os.path.exists(hs_path):
        hs = load_dataset("Rowan/hellaswag", split="validation")
        with open(hs_path, "w") as f:
            for ex in hs:
                f.write(json.dumps({"ctx": ex["ctx"], "endings": ex["endings"],
                                    "label": int(ex["label"])}) + "\n")
        print(f"hellaswag_val: {len(hs)} examples")


if __name__ == "__main__":
    main()
