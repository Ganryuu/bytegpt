"""
Evaluation harness, comparable across input encodings.

    python evaluate.py --ckpt runs/small-10B/ckpt.pt
    python evaluate.py --ckpt runs/bpe-13B/ckpt.pt --tokenizer tokenizer_32k.json
    python evaluate.py --ckpt hf:ShadeEngine/bytegpt-byte-85M --out results/eval_byte.json

--ckpt takes a training checkpoint (.pt), an exported folder (config.json +
model.safetensors, see export_hf.py), or hf:<repo id> to fetch one from the Hub.

Every task normalizes by BYTES of text, never by tokens, so byte-level and
BPE models are scored on the same scale:
  bpb        - bits-per-byte on each frozen domain in eval_data/*.bin
  hellaswag  - 4-way multiple choice; endings scored by per-byte NLL
  typo       - bpb degradation under 1% character corruption
  generate   - qualitative continuations of fixed prompts

An adapter hides the encoding: ByteAdapter feeds raw bytes; BPEAdapter
tokenizes text and converts token NLL back to per-byte bits using each
token's exact byte length (in byte-level BPE, every character of a vocab
entry represents exactly one byte).
"""

import argparse
import glob
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from model import ByteGPT, GPTConfig

LN2 = math.log(2)
DOC_SEP = 254   # byte streams
DOC_TOKEN = 0   # BPE streams

PROMPTS = [
    "The French Revolution began in",
    "def fibonacci(n):\n",
    "Photosynthesis is the process by which",
    "Once upon a time, in a quiet village,",
]


class _HNetShim:
    """Make HNetForCausalLM look like ByteGPT to the shared eval code:
    callable as model(x, y) -> (logits, None)."""

    def __init__(self, hnet):
        self.hnet = hnet

    def __call__(self, x, y=None):
        mask = torch.ones_like(x, dtype=torch.bool)
        return self.hnet(x, mask=mask).logits, None


def read_checkpoint(path, device):
    """Return {"model": state_dict, "config": dict, "iter": int} from a .pt
    checkpoint, an exported folder, or a Hub repo given as hf:<repo id>."""
    if path.startswith("hf:"):
        from huggingface_hub import snapshot_download
        path = snapshot_download(path[3:])
    if os.path.isdir(path):
        from safetensors.torch import load_file
        with open(os.path.join(path, "config.json")) as f:
            meta = json.load(f)
        sd = load_file(os.path.join(path, "model.safetensors"), device=device)
        # tied weights are stored once; point the alias at the same tensor
        for alias, src in meta.get("tied_weights", {}).items():
            sd[alias] = sd[src]
        return {"model": sd, "config": meta["config"], "iter": meta["iter"]}
    return torch.load(path, map_location=device, weights_only=False)


def load_model(ckpt_path, device):
    ck = read_checkpoint(ckpt_path, device)
    if "arch_layout" in ck["config"]:  # H-Net checkpoint (needs .venv-hnet)
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "repos/hnet"))
        from hnet_train import build_model
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(ck["config"], f)
            tmp = f.name
        hnet, _ = build_model(tmp, device)
        os.unlink(tmp)
        hnet.load_state_dict(ck["model"])
        hnet.eval()
        n = sum(p.numel() for p in hnet.parameters())
        # block_size for eval windows; H-Net itself has no fixed context
        cfg = SimpleNamespace(block_size=8192, vocab_size=256, hnet=True)
        print(f"loaded {ckpt_path}: iter {ck['iter']}, {n/1e6:.1f}M params, H-Net")
        return _HNetShim(hnet), cfg, ck["iter"]
    cfg = GPTConfig(**ck["config"])
    model = ByteGPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {ckpt_path}: iter {ck['iter']}, {model.num_params()/1e6:.1f}M params, "
          f"vocab {cfg.vocab_size}")
    return model, cfg, ck["iter"]


@torch.no_grad()
def _window_nll(model, cfg, ids: np.ndarray, byte_len, device, batch_size, max_windows):
    """Sum NLL (nats) and byte-coverage over non-overlapping windows of a
    unit stream. byte_len(targets)->bytes maps predicted units to bytes."""
    T = cfg.block_size
    n_win = min((len(ids) - 1) // T, max_windows)
    total_nats, total_bytes = 0.0, 0
    for start in range(0, n_win, batch_size):
        rows = [ids[i * T: i * T + T + 1] for i in range(start, min(start + batch_size, n_win))]
        buf = torch.from_numpy(np.stack(rows).astype(np.int64)).to(device)
        x, y = buf[:, :-1], buf[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x, y)
        nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                              y.reshape(-1), reduction="sum")
        total_nats += nll.item()
        total_bytes += byte_len(y.cpu().numpy())
    return total_nats, total_bytes


class ByteAdapter:
    def __init__(self, model, cfg, device):
        self.model, self.cfg, self.device = model, cfg, device

    def bpb_on_bytes(self, arr: np.ndarray, batch_size=8, max_windows=256):
        nats, nbytes = _window_nll(self.model, self.cfg, np.asarray(arr),
                                   lambda y: y.size, self.device, batch_size, max_windows)
        return nats / LN2 / nbytes

    @torch.no_grad()
    def ending_nll_per_byte(self, ctx: str, ending: str):
        seq = bytes([DOC_SEP]) + (ctx + ending).encode("utf-8")
        seq = seq[-(self.cfg.block_size + 1):]
        ids = torch.tensor(list(seq), dtype=torch.long, device=self.device)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = self.model(ids[:, :-1], ids[:, 1:])
        logp = F.log_softmax(logits.float(), dim=-1)
        n = len(ending.encode("utf-8"))
        tgt = ids[0, -n:]
        lp = logp[0, -n:].gather(-1, tgt[:, None]).squeeze(-1)
        return -lp.sum().item() / n

    @torch.no_grad()
    def generate(self, prompt: str, max_new=256):
        ids = torch.tensor([DOC_SEP] + list(prompt.encode("utf-8")),
                           dtype=torch.long, device=self.device)[None]
        y = self.model.generate(ids, max_new, temperature=0.8, top_k=50)
        text = bytes(y[0].tolist()[1:]).split(bytes([DOC_SEP]))[0]
        return text.decode("utf-8", errors="replace")


class HNetAdapter(ByteAdapter):
    """Byte-in byte-out like ByteAdapter; only generation differs (the official
    cached-inference path is involved, so we re-forward the full sequence per
    step — slow but simple, and eval only needs a few hundred bytes)."""

    @torch.no_grad()
    def generate(self, prompt: str, max_new=256, temperature=0.8, top_k=50):
        ids = torch.tensor([DOC_SEP] + list(prompt.encode("utf-8")),
                           dtype=torch.long, device=self.device)[None]
        for _ in range(max_new):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = self.model(ids)
            logits = logits[0, -1].float() / temperature
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            if nxt.item() == DOC_SEP:
                break
            ids = torch.cat([ids, nxt[None]], dim=1)
        text = bytes(ids[0].tolist()[1:])
        return text.decode("utf-8", errors="replace")


class BPEAdapter:
    def __init__(self, model, cfg, device, tokenizer_path):
        from tokenizers import Tokenizer
        self.model, self.cfg, self.device = model, cfg, device
        self.tok = Tokenizer.from_file(tokenizer_path)
        # byte length of every vocab id: in ByteLevel BPE each char of the
        # vocab string stands for exactly one byte
        vocab = self.tok.get_vocab()
        self.blen = np.ones(cfg.vocab_size, dtype=np.int64)
        for s, i in vocab.items():
            self.blen[i] = len(s)
        self.blen[DOC_TOKEN] = 1  # <|doc|> counted like the byte separator

    def _byte_stream_to_ids(self, arr: np.ndarray):
        docs = bytes(np.asarray(arr)).split(bytes([DOC_SEP]))
        ids = []
        for d in docs:
            if not d:
                continue
            ids.append(DOC_TOKEN)
            ids.extend(self.tok.encode(d.decode("utf-8", errors="replace")).ids)
        return np.asarray(ids, dtype=np.int64)

    def bpb_on_bytes(self, arr: np.ndarray, batch_size=8, max_windows=256):
        ids = self._byte_stream_to_ids(arr)
        nats, nbytes = _window_nll(self.model, self.cfg, ids,
                                   lambda y: int(self.blen[y].sum()),
                                   self.device, batch_size, max_windows)
        return nats / LN2 / nbytes

    @torch.no_grad()
    def ending_nll_per_byte(self, ctx: str, ending: str):
        # tokenize ctx and ending separately (the standard lm-eval approach;
        # note this sidesteps BPE's prompt-boundary problem imperfectly)
        ids = [DOC_TOKEN] + self.tok.encode(ctx).ids
        end_ids = self.tok.encode(ending).ids
        seq = (ids + end_ids)[-(self.cfg.block_size + 1):]
        t = torch.tensor(seq, dtype=torch.long, device=self.device)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = self.model(t[:, :-1], t[:, 1:])
        logp = F.log_softmax(logits.float(), dim=-1)
        n = len(end_ids)
        tgt = t[0, -n:]
        lp = logp[0, -n:].gather(-1, tgt[:, None]).squeeze(-1)
        return -lp.sum().item() / len(ending.encode("utf-8"))

    @torch.no_grad()
    def generate(self, prompt: str, max_new=64):
        ids = torch.tensor([DOC_TOKEN] + self.tok.encode(prompt).ids,
                           dtype=torch.long, device=self.device)[None]
        y = self.model.generate(ids, max_new, temperature=0.8, top_k=50)
        out = y[0].tolist()[1:]
        if DOC_TOKEN in out:
            out = out[:out.index(DOC_TOKEN)]
        return self.tok.decode(out)


def task_bpb(adapter):
    out = {}
    for path in sorted(glob.glob("eval_data/*.bin")):
        name = os.path.basename(path)[:-4]
        arr = np.memmap(path, dtype=np.uint8, mode="r")
        out[f"bpb/{name}"] = adapter.bpb_on_bytes(arr)
    return out


def task_hellaswag(adapter, limit=1000):
    correct = total = 0
    with open("eval_data/hellaswag_val.jsonl") as f:
        for line in f:
            ex = json.loads(line)
            scores = [adapter.ending_nll_per_byte(ex["ctx"], " " + e)
                      for e in ex["endings"]]
            correct += int(np.argmin(scores) == int(ex["label"]))
            total += 1
            if total >= limit:
                break
    return {"hellaswag/acc_norm": correct / total, "hellaswag/n": total}


def task_typo(adapter, corrupt_frac=0.01, seed=0):
    arr = np.array(np.memmap("eval_data/fineweb_val.bin", dtype=np.uint8, mode="r"))
    clean = adapter.bpb_on_bytes(arr, max_windows=64)
    rng = np.random.default_rng(seed)
    idx = np.where((arr >= 97) & (arr <= 122))[0]
    hit = rng.choice(idx, size=int(len(idx) * corrupt_frac), replace=False)
    arr[hit] = rng.integers(97, 123, size=len(hit)).astype(np.uint8)
    corrupted = adapter.bpb_on_bytes(arr, max_windows=64)
    return {"typo/clean_bpb": clean, "typo/corrupt_bpb": corrupted,
            "typo/ratio": corrupted / clean}


def task_generate(adapter, max_new_bytes=256):
    # for BPE, max_new is in tokens: convert so both emit ~equal text
    n = max_new_bytes if isinstance(adapter, ByteAdapter) else max_new_bytes // 4
    return {p: adapter.generate(p, n) for p in PROMPTS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer json -> evaluate as a BPE model")
    ap.add_argument("--tasks", default="bpb,hellaswag,typo,generate")
    ap.add_argument("--hellaswag-limit", type=int, default=1000)
    ap.add_argument("--out", default=None, help="also write results to this json file")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="bytegpt")
    args = ap.parse_args()

    device = "cuda"
    model, cfg, it = load_model(args.ckpt, device)
    if args.tokenizer:
        adapter = BPEAdapter(model, cfg, device, args.tokenizer)
    elif getattr(cfg, "hnet", False):
        adapter = HNetAdapter(model, cfg, device)
    else:
        adapter = ByteAdapter(model, cfg, device)

    tasks = args.tasks.split(",")
    results = {}
    if "bpb" in tasks:
        results.update(task_bpb(adapter))
    if "hellaswag" in tasks:
        results.update(task_hellaswag(adapter, args.hellaswag_limit))
    if "typo" in tasks:
        results.update(task_typo(adapter))

    print("\n=== results ===")
    for k, v in results.items():
        print(f"{k:28s} {v:.4f}" if isinstance(v, float) else f"{k:28s} {v}")

    samples = {}
    if "generate" in tasks:
        samples = task_generate(adapter)
        for p, s in samples.items():
            print(f"\n--- {p!r}\n{s}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"ckpt": args.ckpt, "iter": it, "results": results,
                       "samples": samples}, f, indent=2, ensure_ascii=False)

    if args.wandb:
        import wandb
        run_name = f"eval-{os.path.basename(os.path.dirname(args.ckpt))}-it{it}"
        wandb.init(project=args.wandb_project, name=run_name, job_type="eval",
                   config={"ckpt": args.ckpt, "iter": it, "tokenizer": args.tokenizer})
        wandb.log(results)
        if samples:
            wandb.log({"samples": wandb.Table(
                columns=["prompt", "completion"], data=list(samples.items()))})
        wandb.finish()


if __name__ == "__main__":
    main()
