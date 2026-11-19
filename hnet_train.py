"""
Train a 1-stage H-Net on byte shards.

Differences from train.py (the isotropic trainer):
  - model: HNetForCausalLM from the official goombalab/hnet repo
  - loss: next-byte CE + ratio_weight * load-balancing loss per stage
    (keeps the learned chunker near the target compression N without
    dictating WHERE to cut)
  - per-stage LR multipliers (outer byte-rate stages train hotter)
  - no torch.compile (dynamic chunk lengths break graph capture)
  - logs chunking telemetry: achieved compression ratio and boundary prob

Usage:
  python hnet_train.py --config configs/hnet_micro.json --max-iters 50 --no-wandb
  torchrun --standalone --nproc_per_node=2 hnet_train.py --config configs/hnet_small.json
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "repos/hnet"))

from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig
from hnet.models.mixer_seq import HNetForCausalLM
from hnet.utils.train import load_balancing_loss, group_params

from data import ByteShards
from train import lr_at

LN2 = math.log(2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="hnet config json")
    p.add_argument("--data-dir", default="data/fineweb_edu_bytes")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--block-size", type=int, default=8192)
    p.add_argument("--micro-bs", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-iters", type=int, default=20000)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--lr-mult", default="2.0,1.0",
                   help="per-stage LR multipliers, outer first")
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--warmup-iters", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--target-n", type=float, default=4.6,
                   help="target bytes per chunk for the ratio loss")
    p.add_argument("--ratio-weight", type=float, default=0.03)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-windows", type=int, default=64)
    p.add_argument("--ckpt-every", type=int, default=1000,
                   help="overwrite the resumable ckpt.pt this often")
    p.add_argument("--snap-every", type=int, default=8000,
                   help="keep a model-only snapshot this often")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb-project", default="bytegpt")
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def build_model(config_path, device):
    with open(config_path) as f:
        raw = json.load(f)
    cfg = HNetConfig(
        arch_layout=raw["arch_layout"],
        d_model=raw["d_model"],
        d_intermediate=raw["d_intermediate"],
        vocab_size=raw.get("vocab_size", 256),
        ssm_cfg=SSMConfig(**raw.get("ssm_cfg", {})),
        attn_cfg=AttnConfig(**raw.get("attn_cfg", {})),
        tie_embeddings=raw.get("tie_embeddings", True),
    )
    model = HNetForCausalLM(cfg, device=device)
    model.init_weights()
    return model, cfg


def main():
    args = parse_args()
    lr_mult = [float(x) for x in args.lr_mult.split(",")]

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
        torch.cuda.set_device(device)
    else:
        rank, world = 0, 1
        device = "cuda"
    master = rank == 0

    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    model, cfg = build_model(args.config, device)
    n_params = sum(p.numel() for p in model.parameters())
    n_stages = len(cfg.d_model)
    assert len(lr_mult) == n_stages, f"need {n_stages} lr multipliers"

    name = os.path.splitext(os.path.basename(args.config))[0]
    out_dir = args.out_dir or f"runs/{name}"
    if master:
        os.makedirs(out_dir, exist_ok=True)
        print(f"config={name}  params={n_params/1e6:.1f}M  stages={n_stages}  "
              f"target N={args.target_n}")

    # per-stage LR multipliers -> param groups (hnet's own machinery)
    model.apply_lr_multiplier(lr_mult)
    groups = group_params(model)
    for g in groups:
        g.setdefault("weight_decay", args.weight_decay)
        g["lr_multiplier"] = g.pop("lr_multiplier", 1.0)
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), fused=True)

    start_iter = 0
    wandb_id = None
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_iter = ck["iter"] + 1
        wandb_id = ck.get("wandb_id")
        del ck
        if master:
            print(f"resumed from {args.resume} at iter {start_iter}")

    if ddp:
        model = DDP(model, device_ids=[torch.cuda.current_device()])
    raw_model = model.module if ddp else model

    train_data = ByteShards(args.data_dir, "train")
    val_data = ByteShards(args.data_dir, "val")
    # start_iter in the seed: a resumed run must not replay the first crops
    rng = np.random.default_rng(args.seed + 10_000 * rank + start_iter)

    if master and args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, id=wandb_id, resume="allow",
                   name=args.run_name or f"{name}-{time.strftime('%m%d-%H%M')}",
                   config={**vars(args), "n_params": n_params, "world_size": world,
                           "arch_layout": cfg.arch_layout, "d_model": cfg.d_model})
        wandb_id = wandb.run.id

    def forward_loss(x, y):
        """CE + ratio losses. Returns (total, ce, achieved compression)."""
        mask = torch.ones_like(x, dtype=torch.bool)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, mask=mask)
        logits = out.logits.float()
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        ratio = 0.0
        for bp in out.bpred_output:  # one per hierarchy stage
            ratio = ratio + load_balancing_loss(bp, args.target_n)
        # telemetry: fraction of positions chosen as boundaries at stage 0
        comp = 1.0 / max(out.bpred_output[0].boundary_mask.float().mean().item(), 1e-6)
        return ce + args.ratio_weight * ratio, ce, comp

    @torch.no_grad()
    def eval_val():
        raw_model.eval()
        losses, comps = [], []
        for win in val_data.sequential_windows(args.block_size, args.eval_windows):
            buf = torch.from_numpy(win.astype(np.int64))[None].to(device)
            _, ce, comp = forward_loss(buf[:, :-1], buf[:, 1:])
            losses.append(ce.item())
            comps.append(comp)
        raw_model.train()
        return float(np.mean(losses)) / LN2, float(np.mean(comps))

    bytes_per_step = args.micro_bs * args.block_size * args.grad_accum * world
    model.train()
    t0 = time.time()
    run_ce, run_comp, run_n = 0.0, 0.0, 0
    for it in range(start_iter, args.max_iters):
        lr = lr_at(it, args)
        for g in opt.param_groups:
            g["lr"] = lr * g.get("lr_multiplier", 1.0)

        for micro in range(args.grad_accum):
            x, y = train_data.sample_batch(args.micro_bs, args.block_size, rng, device)
            if ddp:
                model.require_backward_grad_sync = micro == args.grad_accum - 1
            total, ce, comp = forward_loss(x, y)
            (total / args.grad_accum).backward()
            run_ce += ce.item() / args.grad_accum
            run_comp += comp / args.grad_accum

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        run_n += 1

        if master and (it % args.log_every == 0 or it == args.max_iters - 1):
            dt = time.time() - t0
            t0 = time.time()
            steps = run_n
            bps = bytes_per_step * steps / dt
            avg_ce, avg_comp = run_ce / steps, run_comp / steps
            run_ce, run_comp, run_n = 0.0, 0.0, 0
            print(f"iter {it:6d}  ce {avg_ce:.4f}  bpb {avg_ce/LN2:.4f}  "
                  f"N {avg_comp:.2f}  lr {lr:.2e}  gnorm {grad_norm:.2f}  "
                  f"{bps/1e3:.0f} KB/s", flush=True)
            if args.wandb:
                import wandb
                wandb.log({"train/bpb": avg_ce / LN2, "train/compression": avg_comp,
                           "lr": lr, "grad_norm": grad_norm.item(),
                           "throughput_bytes_s": bps,
                           "bytes_seen": (it + 1) * bytes_per_step}, step=it)

        # shards may still be arriving on disk; pick up new ones (every rank)
        if it and it % args.eval_every == 0:
            n_before = len(train_data.shards)
            train_data = ByteShards(args.data_dir, "train")
            if master and len(train_data.shards) != n_before:
                print(f"train shards: {n_before} -> {len(train_data.shards)}", flush=True)

        if it and it % args.eval_every == 0 and master:
            vbpb, vcomp = eval_val()
            print(f"iter {it:6d}  VAL bpb {vbpb:.4f}  N {vcomp:.2f}", flush=True)
            if args.wandb:
                import wandb
                wandb.log({"val/bpb": vbpb, "val/compression": vcomp}, step=it)

        if master and it and (it % args.ckpt_every == 0 or it == args.max_iters - 1):
            ck = {"model": raw_model.state_dict(), "opt": opt.state_dict(),
                  "iter": it, "config": json.load(open(args.config)),
                  "args": vars(args), "wandb_id": wandb_id}
            # write-then-rename so a crash mid-save can't corrupt the only ckpt
            tmp = os.path.join(out_dir, "ckpt.pt.tmp")
            torch.save(ck, tmp)
            os.replace(tmp, os.path.join(out_dir, "ckpt.pt"))
            if it % args.snap_every == 0 or it == args.max_iters - 1:
                torch.save({"model": raw_model.state_dict(), "iter": it,
                            "config": json.load(open(args.config))},
                           os.path.join(out_dir, f"snap_{it:06d}.pt"))
            print(f"saved checkpoint at iter {it}", flush=True)

    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"rank {rank}: peak GPU mem {peak:.2f} GiB", flush=True)
    if master:
        vbpb, vcomp = eval_val()
        print(f"FINAL val bpb {vbpb:.4f}  compression N {vcomp:.2f}")
        if args.wandb:
            import wandb
            wandb.log({"val/bpb": vbpb, "val/compression": vcomp})
            wandb.finish()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
