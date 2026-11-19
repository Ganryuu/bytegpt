"""
Train a 32k byte-level BPE tokenizer on our own FineWeb-Edu byte shards.

This is the "offline compression stage" of the classic LLM pipeline: BPE
greedily merges the most frequent adjacent pair until the vocab is full.
ByteLevel pre-tokenization = GPT-2's scheme: text is first split by a
regex (so merges cannot cross word boundaries -- the hand-engineered
"pretokenization" prior the Cartesia posts measure), and operates on bytes,
so ANY input is representable (no <unk>).

Trained on a 2 GB sample: BPE statistics saturate quickly; GPT-2's own
tokenizer was trained on a similar order of magnitude.

Output: tokenizer_32k.json  (+ compression stats on held-out val data)
"""

import numpy as np
from tokenizers import ByteLevelBPETokenizer

DOC_SEP = 254
VOCAB = 32768
SAMPLE_BYTES = 2_000_000_000
TRAIN_SHARDS = ["data/fineweb_edu_bytes/train_00000.bin",
                "data/fineweb_edu_bytes/train_00001.bin",
                "data/fineweb_edu_bytes/train_00002.bin",
                "data/fineweb_edu_bytes/train_00003.bin"]


def iter_docs(paths, max_bytes):
    total = 0
    for p in paths:
        arr = np.memmap(p, dtype=np.uint8, mode="r")
        seps = np.where(arr == DOC_SEP)[0]
        for i in range(len(seps) - 1):
            doc = bytes(arr[seps[i] + 1: seps[i + 1]]).decode("utf-8", errors="replace")
            yield doc
            total += len(doc)
            if total >= max_bytes:
                return


def main():
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(
        iter_docs(TRAIN_SHARDS, SAMPLE_BYTES),
        vocab_size=VOCAB,
        min_frequency=2,
        special_tokens=["<|doc|>"],
        show_progress=True,
    )
    tok.save("tokenizer_32k.json")

    # measure compression on held-out val (never seen by the tokenizer)
    val_docs = list(iter_docs(["data/fineweb_edu_bytes/val_00000.bin"], 20_000_000))
    n_bytes = sum(len(d.encode("utf-8")) for d in val_docs)
    n_tokens = sum(len(e) for e in tok.encode_batch(val_docs))
    print(f"vocab={VOCAB}  val sample={n_bytes/1e6:.0f} MB")
    print(f"compression: {n_bytes/n_tokens:.3f} bytes/token")


if __name__ == "__main__":
    main()
