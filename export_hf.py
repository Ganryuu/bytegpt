"""
Export the three trained models to Hugging Face format and (optionally) upload.

    python export_hf.py                 # write export/<name>/ for each model
    python export_hf.py --upload        # also push each folder to the Hub

Each exported folder holds:
  model.safetensors  final weights, float32, no optimizer state
  config.json        architecture config, training arguments, final iteration
  README.md          model card (results come from results/eval_<key>.json)
  model.py           the model code (byte and BPE models)
  tokenizer_32k.json the tokenizer (BPE model)

evaluate.py and chunk_viz.py read these folders directly (--ckpt export/<name>).
"""

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file

from model_cards import card

MODELS = {
    "byte": dict(run="runs/small-10B", name="bytegpt-byte-85M", arch="ByteGPT",
                 files=["model.py"]),
    "bpe": dict(run="runs/bpe-13B", name="bytegpt-bpe32k-110M", arch="ByteGPT",
                files=["model.py", "tokenizer_32k.json"]),
    "hnet": dict(run="runs/hnet-44B", name="bytegpt-hnet-100M", arch="HNet",
                 files=[]),
}


def convert(key, spec, out_dir):
    ck = torch.load(os.path.join(spec["run"], "ckpt.pt"), map_location="cpu",
                    weights_only=False)
    sd = ck["model"]
    # safetensors stores every tensor once: keep the first name of each
    # shared tensor and record the others so the loader can re-tie them
    seen, tied, out = {}, {}, {}
    for k, v in sd.items():
        ptr = v.data_ptr()
        if ptr in seen:
            tied[k] = seen[ptr]
        else:
            seen[ptr] = k
            out[k] = v.contiguous()
    n_params = sum(v.numel() for v in out.values())
    save_file(out, os.path.join(out_dir, "model.safetensors"))
    drop = {"out_dir", "resume", "wandb", "wandb_project", "run_name", "data_dir"}
    meta = {
        "architecture": spec["arch"],
        "config": ck["config"],
        "iter": ck["iter"],
        "n_params": n_params,
        "tied_weights": tied,
        "training_args": {k: v for k, v in ck["args"].items() if k not in drop},
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{key}: {n_params/1e6:.1f}M params, iter {ck['iter']}, tied {tied}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="ShadeEngine")
    ap.add_argument("--only", default=None, help="byte | bpe | hnet")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--code-url", default=None, help="GitHub repo to link in the cards")
    args = ap.parse_args()

    for key, spec in MODELS.items():
        if args.only and key != args.only:
            continue
        out_dir = os.path.join("export", spec["name"])
        os.makedirs(out_dir, exist_ok=True)
        if not os.path.exists(os.path.join(out_dir, "model.safetensors")):
            convert(key, spec, out_dir)
        for f in spec["files"]:
            shutil.copy(f, out_dir)
        with open(os.path.join(out_dir, "README.md"), "w") as f:
            f.write(card(key, args.user, args.code_url))

        if args.upload:
            from huggingface_hub import HfApi
            api = HfApi()
            repo = f"{args.user}/{spec['name']}"
            api.create_repo(repo, repo_type="model", exist_ok=True)
            api.upload_folder(repo_id=repo, repo_type="model", folder_path=out_dir)
            print(f"uploaded: https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
