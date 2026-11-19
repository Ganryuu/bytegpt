"""
Model cards for the three Hugging Face repos. export_hf.py calls card().

The results table is filled from results/eval_<key>.json, which evaluate.py
writes with --out. Nothing here is typed in by hand except the prose.
"""

import json
import os

ARTICLE = "https://claude.ai/artifact/AamTLQSJGp6uYueALTKjtm"
HNET_REPO = "https://github.com/goombalab/hnet"
HNET_COMMIT = "3673fe1"

NAMES = {
    "byte": "bytegpt-byte-85M",
    "bpe": "bytegpt-bpe32k-110M",
    "hnet": "bytegpt-hnet-100M",
}
LABELS = {"byte": "Byte Transformer", "bpe": "BPE Transformer", "hnet": "H-Net"}

ROWS = [
    ("bpb/fineweb_val", "FineWeb-Edu validation, bits per byte", "{:.3f}"),
    ("bpb/wiki_en", "English Wikipedia, bits per byte", "{:.3f}"),
    ("bpb/wiki_fr", "French Wikipedia, bits per byte", "{:.3f}"),
    ("bpb/code_py", "Python code, bits per byte", "{:.3f}"),
    ("hellaswag/acc_norm", "HellaSwag accuracy (1,000 questions, chance is 25%)", "{:.1%}"),
    ("typo/ratio", "Typo penalty (loss with 1% typos / clean loss)", "{:.3f}"),
]


def results_table(user, me=None):
    res = {}
    for key in NAMES:
        with open(os.path.join("results", f"eval_{key}.json")) as f:
            res[key] = json.load(f)["results"]

    def head(key):
        label = f"[{LABELS[key]}](https://huggingface.co/{user}/{NAMES[key]})"
        return f"**{label} (this model)**" if key == me else label

    lines = ["| | " + " | ".join(head(k) for k in NAMES) + " |",
             "|---|" + "---:|" * len(NAMES)]
    for metric, label, fmt in ROWS:
        cells = []
        for key in NAMES:
            cell = fmt.format(res[key][metric])
            cells.append(f"**{cell}**" if key == me else cell)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


FRONT = """---
language:
- en
pipeline_tag: text-generation
tags:
{tags}
datasets:
- {dataset}
---
"""

SHARED_TAIL = """
## Results

All three models were trained with the same compute budget, about 1.0e19
floating-point operations each, on the same documents. Lower is better for
every row except HellaSwag.

{table}

Bits per byte is the model's loss divided by the bytes of text it covers, so
models with different tokenizers can be compared on one scale. Each test set
is 5 MB of text that none of the models trained on.

## Limits

- One training run per model. Differences of a percent or two could be noise.
- These are small research models trained on educational web text. They write
  fluent English that is often factually wrong. Do not use them for anything
  beyond studying how such models behave.
- The H-Net settled at 3.7 bytes per chunk, shorter than the tokenizer's 4.6.
  Its main network therefore ran more steps per byte and saw less text
  (38 GB against 60 GB for the BPE model) for the same compute.

## More

- Article: [Can a model learn its own tokenizer?]({article})
{code_line}- Training data: [{user}/fineweb-edu-bytes-10BT](https://huggingface.co/datasets/{user}/fineweb-edu-bytes-10BT)
  and [{user}/fineweb-edu-bpe32k-10BT](https://huggingface.co/datasets/{user}/fineweb-edu-bpe32k-10BT)
- The experiment follows Cartesia's
  [From Tokens to Bytes](https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1) posts
  and the [H-Net paper](https://arxiv.org/abs/2507.07955).
"""

BYTE = """
# Byte Transformer, 85M parameters

A language model that reads and writes raw bytes. It has no tokenizer: every
UTF-8 byte of the text is one step. It is the baseline in a three-way
comparison of byte, BPE and H-Net models at equal compute.

## Model

- 12-layer Transformer, 12 heads, width 768, 85.2M parameters
- Vocabulary of 256 byte values, context of 8,192 bytes
- RMSNorm, rotary position embeddings, SwiGLU, tied input and output embeddings
- Byte 254 marks the start of every document

## Training

- Data: FineWeb-Edu `sample-10BT` as raw bytes
- 20,000 steps of 524,288 bytes: 10.5 GB of text, about 1.0e19 FLOPs
- AdamW, peak learning rate 6e-4, 500 warmup steps, cosine decay to 6e-5
- Weight decay 0.1, gradient clipping 1.0, bfloat16, two RTX 3090s for about a day
- The full argument list is in `config.json`

## Use

`model.py` in this repo holds the whole model. It needs `torch`,
`safetensors` and `huggingface_hub`.

```python
import json, sys, torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

path = snapshot_download("{user}/{name}")
sys.path.insert(0, path)
from model import ByteGPT, GPTConfig

meta = json.load(open(f"{{path}}/config.json"))
weights = load_file(f"{{path}}/model.safetensors")
for alias, source in meta["tied_weights"].items():  # tied embeddings are stored once
    weights[alias] = weights[source]
model = ByteGPT(GPTConfig(**meta["config"]))
model.load_state_dict(weights)
model.eval().cuda()

prompt = bytes([254]) + "The French Revolution began in".encode("utf-8")
ids = torch.tensor(list(prompt), device="cuda")[None]
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model.generate(ids, max_new_bytes=200, temperature=0.8, top_k=50)
print(bytes(out[0, 1:].tolist()).decode("utf-8", errors="replace"))
```
"""

BPE = """
# BPE Transformer, 110M parameters

A standard token-based language model. It is the same Transformer as the
[byte model](https://huggingface.co/{user}/bytegpt-byte-85M), but it reads
text through a 32,768-entry BPE tokenizer. It is the tokenizer baseline in a
three-way comparison of byte, BPE and H-Net models at equal compute.

## Model

- 12-layer Transformer, 12 heads, width 768, 110.1M parameters
- Vocabulary of 32,768 tokens, context of 2,048 tokens (about 9,400 bytes)
- RMSNorm, rotary position embeddings, SwiGLU, tied input and output embeddings
- `tokenizer_32k.json` is a byte-level BPE tokenizer trained on 2 GB of the
  same corpus. It averages 4.60 bytes per token. Token 0 starts every document.

## Training

- Data: FineWeb-Edu `sample-10BT`, tokenized
- 25,000 steps of 524,288 tokens: 13.1B tokens, about 60 GB of text and
  1.3 passes over the corpus, for about 1.0e19 FLOPs
- AdamW, peak learning rate 6e-4, 500 warmup steps, cosine decay to 6e-5
- Weight decay 0.1, gradient clipping 1.0, bfloat16, two RTX 3090s for about a day
- The full argument list is in `config.json`

## Use

`model.py` in this repo holds the whole model. It needs `torch`,
`tokenizers`, `safetensors` and `huggingface_hub`.

```python
import json, sys, torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from tokenizers import Tokenizer

path = snapshot_download("{user}/{name}")
sys.path.insert(0, path)
from model import ByteGPT, GPTConfig

meta = json.load(open(f"{{path}}/config.json"))
weights = load_file(f"{{path}}/model.safetensors")
for alias, source in meta["tied_weights"].items():  # tied embeddings are stored once
    weights[alias] = weights[source]
model = ByteGPT(GPTConfig(**meta["config"]))  # the class name is historical
model.load_state_dict(weights)
model.eval().cuda()

tok = Tokenizer.from_file(f"{{path}}/tokenizer_32k.json")
ids = torch.tensor([0] + tok.encode("The French Revolution began in").ids, device="cuda")[None]
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model.generate(ids, max_new_bytes=60, temperature=0.8, top_k=50)  # counts tokens here
print(tok.decode(out[0, 1:].tolist()))
```
"""

HNET = """
# H-Net, 100M parameters

A language model that reads raw bytes and learns where to cut them into
chunks. No tokenizer is involved. A small network reads every byte and marks
chunk boundaries, a large Transformer runs once per chunk, and a small network
turns the result back into byte predictions. It is the learned-chunking model
in a three-way comparison of byte, BPE and H-Net models at equal compute.

## Model

- One-stage H-Net, 99.8M parameters, layout `["m4", ["T12"], "m4"]`
- Encoder and decoder: 4 Mamba-2 layers each, width 512, one step per byte
- Main network: 12-layer Transformer, width 768, one step per chunk
- Vocabulary of 256 byte values. Byte 254 marks the start of every document.
- Learned chunk size: 3.8 bytes on the training distribution, 2.3 on French,
  1.8 on Python code

## Training

- Data: FineWeb-Edu `sample-10BT` as raw bytes, in windows of 8,192 bytes
- 72,800 steps of 524,288 bytes: 38.2 GB of text, about 1.0e19 FLOPs
- Loss: next-byte cross-entropy plus 0.03 times the paper's ratio loss, with a
  target of 4.6 bytes per chunk
- AdamW, peak learning rate 6e-4 (doubled for the byte-level networks),
  2,000 warmup steps, cosine decay
- Weight decay 0.1, gradient clipping 1.0, bfloat16, two RTX 3090s for about
  two and a half days
- The full argument list is in `config.json`

## Use

The model runs on the official H-Net code, which is not on PyPI. It needs a
CUDA GPU, `mamba-ssm` and `flash-attn`.

```bash
git clone {hnet_repo} && cd hnet
git checkout {hnet_commit}
pip install -e .
pip install safetensors huggingface_hub
```

```python
import json, torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig
from hnet.models.mixer_seq import HNetForCausalLM

path = snapshot_download("{user}/{name}")
raw = json.load(open(f"{{path}}/config.json"))["config"]
config = HNetConfig(
    arch_layout=raw["arch_layout"], d_model=raw["d_model"],
    d_intermediate=raw["d_intermediate"], vocab_size=raw["vocab_size"],
    ssm_cfg=SSMConfig(**raw["ssm_cfg"]), attn_cfg=AttnConfig(**raw["attn_cfg"]),
    tie_embeddings=raw["tie_embeddings"],
)
model = HNetForCausalLM(config, device="cuda")
model.load_state_dict(load_file(f"{{path}}/model.safetensors", device="cuda"))
model.eval()

text = bytes([254]) + "The French Revolution began in 1789".encode("utf-8")
ids = torch.tensor(list(text), device="cuda")[None]
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(ids, mask=torch.ones_like(ids, dtype=torch.bool))

next_byte = out.logits[0, -1].float().argmax().item()   # the model's guess for the next byte
starts = out.bpred_output[0].boundary_mask[0].tolist()  # True where a new chunk starts
print(repr(chr(next_byte)))
print("".join(("|" if s else "") + chr(b) for b, s in zip(text[1:], starts[1:])))
```
"""

SPECS = {
    "byte": (BYTE, ["byte-level", "tokenizer-free", "language-modeling"],
             "fineweb-edu-bytes-10BT"),
    "bpe": (BPE, ["bpe", "language-modeling"], "fineweb-edu-bpe32k-10BT"),
    "hnet": (HNET, ["byte-level", "tokenizer-free", "h-net", "mamba", "language-modeling"],
             "fineweb-edu-bytes-10BT"),
}


def card(key, user, code_url=None):
    body, tags, dataset = SPECS[key]
    code_line = f"- Training and evaluation code: [{code_url}]({code_url})\n" if code_url else ""
    front = FRONT.format(tags="\n".join(f"- {t}" for t in tags), dataset=f"{user}/{dataset}")
    body = body.format(user=user, name=NAMES[key], hnet_repo=HNET_REPO, hnet_commit=HNET_COMMIT)
    tail = SHARED_TAIL.format(table=results_table(user, key), article=ARTICLE,
                              code_line=code_line, user=user)
    return front + body + tail
