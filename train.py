"""
Train ByteGPT. Single GPU:
    python train.py --preset micro --max-iters 50 --no-wandb
Both 3090s (DDP):
    torchrun --standalone --nproc_per_node=2 train.py --preset small

Key quantities:
  - loss is mean cross-entropy in nats/byte; bpb = loss / ln(2).
    bits-per-byte is comparable across ANY input encoding (BPE, bytes,
    H-Net), which is why it is the metric of this whole project.
  - effective batch = micro_bs * block_size * grad_accum * world_size bytes.
    Gradient accumulation simulates a large batch on small GPUs: run N
    micro-batches, sum gradients, step once.
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data import ByteShards
from model import PRESETS, ByteGPT, GPTConfig

LN2 = math.log(2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="small", choices=list(PRESETS))
    p.add_argument("--data-dir", default="data/fineweb_edu_bytes")
    p.add_argument("--out-dir", default=None, help="default: runs/<preset>")
    p.add_argument("--micro-bs", type=int, default=4, help="sequences per GPU per fwd/bwd")
    p.add_argument("--grad-accum", type=int, default=8, help="micro-steps per optimizer step")
    p.add_argument("--max-iters", type=int, default=20000)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--warmup-iters", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-windows", type=int, default=64, help="val windows per eval")
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb-project", default="bytegpt")
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None, help="path to checkpoint to resume from")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def lr_at(it, args):
    """Warmup then cosine decay to min_lr. Standard pretraining schedule."""
    if it < args.warmup_iters:
        return args.lr * (it + 1) / args.warmup_iters
    t = (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
    coef = 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
    min_lr = args.lr * args.min_lr_frac
    return min_lr + coef * (args.lr - min_lr)


def main():
    args = parse_args()

    # ---- distributed setup: torchrun sets RANK/WORLD_SIZE env vars ----
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
        torch.cuda.set_device(device)
    else:
        rank, world = 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    master = rank == 0

    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = GPTConfig(**PRESETS[args.preset])
    model = ByteGPT(cfg).to(device)
    n_params = model.num_params()

    out_dir = args.out_dir or f"runs/{args.preset}"
    if master:
        os.makedirs(out_dir, exist_ok=True)

    train_data = ByteShards(args.data_dir, "train")
    val_data = ByteShards(args.data_dir, "val")
    # for BPE data a "unit" is a token covering ~4.6 bytes; for byte data
    # unit == byte. Loss is per unit; bpb = loss / ln2 / unit_bytes.
    unit_bytes = train_data.meta.get("avg_bytes_per_token", 1.0)

    units_per_step = args.micro_bs * cfg.block_size * args.grad_accum * world
    # FLOPs/unit ~ 6N (fwd+bwd) + causal attention term 6*L*T*d
    # (causal flash attention computes only the lower triangle, so half of
    # the full 12*L*T*d convention)
    flops_per_unit = 6 * n_params + 6 * cfg.n_layer * cfg.block_size * cfg.n_embd
    if master:
        print(f"preset={args.preset}  params={n_params/1e6:.1f}M  "
              f"batch={units_per_step/1e3:.0f}K units/step ({unit_bytes:.2f} B/unit)  "
              f"total={args.max_iters * units_per_step / 1e9:.1f}B units, "
              f"{args.max_iters * units_per_step * flops_per_unit:.2e} FLOPs planned")

    opt = model.configure_optimizer(args.weight_decay, args.lr, (0.9, 0.95), "cuda")

    start_iter = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_iter = ck["iter"] + 1
        if master:
            print(f"resumed from {args.resume} at iter {start_iter}")

    if args.compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[torch.cuda.current_device()])
    raw = model.module if ddp else model
    raw_uncompiled = getattr(raw, "_orig_mod", raw)

    rng = np.random.default_rng(args.seed + 10_000 * rank)

    if master and args.wandb:
        import wandb
        wandb.init(project=args.wandb_project,
                   name=args.run_name or f"{args.preset}-{time.strftime('%m%d-%H%M')}",
                   config={**vars(args), **cfg.__dict__, "n_params": n_params,
                           "world_size": world, "units_per_step": units_per_step,
                           "unit_bytes": unit_bytes})

    @torch.no_grad()
    def eval_val_bpb():
        raw.eval()
        losses = []
        for win in val_data.sequential_windows(cfg.block_size, args.eval_windows):
            buf = torch.from_numpy(win.astype(np.int64))[None].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = raw(buf[:, :-1], buf[:, 1:])
            losses.append(loss.item())
        raw.train()
        return float(np.mean(losses)) / LN2 / unit_bytes

    model.train()
    t0 = time.time()
    running_loss = 0.0
    for it in range(start_iter, args.max_iters):
        lr = lr_at(it, args)
        for g in opt.param_groups:
            g["lr"] = lr

        for micro in range(args.grad_accum):
            x, y = train_data.sample_batch(args.micro_bs, cfg.block_size, rng, device)
            if ddp:  # sync gradients only on the last micro-step
                model.require_backward_grad_sync = micro == args.grad_accum - 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            (loss / args.grad_accum).backward()
            running_loss += loss.item() / args.grad_accum

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if master and (it % args.log_every == 0 or it == args.max_iters - 1):
            dt = time.time() - t0
            t0 = time.time()
            steps = args.log_every if it else 1
            ups = units_per_step * steps / dt
            bps = ups * unit_bytes
            mfu = ups * flops_per_unit / (world * 71e12)  # 3090 bf16 peak ~71 TFLOPS
            avg = running_loss / steps
            running_loss = 0.0
            print(f"iter {it:6d}  loss {avg:.4f}  bpb {avg/LN2/unit_bytes:.4f}  lr {lr:.2e}  "
                  f"gnorm {grad_norm:.2f}  {bps/1e3:.0f} KB/s  mfu {mfu*100:.1f}%", flush=True)
            if args.wandb:
                import wandb
                wandb.log({"train/loss": avg, "train/bpb": avg / LN2 / unit_bytes, "lr": lr,
                           "grad_norm": grad_norm.item(), "throughput_bytes_s": bps,
                           "mfu": mfu, "bytes_seen": (it + 1) * units_per_step * unit_bytes}, step=it)

        if it and it % args.eval_every == 0 and master:
            vbpb = eval_val_bpb()
            print(f"iter {it:6d}  VAL bpb {vbpb:.4f}", flush=True)
            if args.wandb:
                import wandb
                wandb.log({"val/bpb": vbpb}, step=it)

        if master and it and (it % args.ckpt_every == 0 or it == args.max_iters - 1):
            ck = {"model": raw_uncompiled.state_dict(), "opt": opt.state_dict(),
                  "iter": it, "config": cfg.__dict__, "args": vars(args)}
            path = os.path.join(out_dir, "ckpt.pt")
            torch.save(ck, path)
            # model-only snapshot (no optimizer state): 4 bytes/param instead
            # of 12, kept per-iteration for the training-progression evals
            snap = os.path.join(out_dir, f"snap_{it:06d}.pt")
            torch.save({"model": raw_uncompiled.state_dict(), "iter": it,
                        "config": cfg.__dict__}, snap)
            print(f"saved {path} + {snap}", flush=True)

    if master:
        vbpb = eval_val_bpb()
        print(f"FINAL val bpb {vbpb:.4f}")
        if args.wandb:
            import wandb
            wandb.log({"val/bpb": vbpb})
            wandb.finish()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
