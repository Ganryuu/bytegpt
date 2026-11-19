"""
Byte-level GPT. nanoGPT-style minimalism, modern internals:
  - vocab = 256 (raw bytes). No tokenizer anywhere.
  - RMSNorm (pre-norm), RoPE, SwiGLU MLP, no biases
  - flash attention via F.scaled_dot_product_attention
  - weight tying between byte embedding and output head

The byte embedding table is 256 x d_model -- essentially free. In a BPE
model, the 50k x d_model embedding can be a third of all parameters; here
virtually all parameters do actual computation.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 256
    block_size: int = 8192  # context length in bytes
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    rope_theta: float = 100000.0


PRESETS = {
    # ~10M params: smoke tests and pipeline debugging
    "micro": dict(n_layer=6, n_head=6, n_embd=384, block_size=2048),
    # ~29M params: scaling-ladder point
    "mini": dict(n_layer=9, n_head=9, n_embd=576, block_size=8192),
    # ~85M non-embedding params: GPT-2 small compute equivalent
    "small": dict(n_layer=12, n_head=12, n_embd=768, block_size=8192),
    # BPE twin of "small": same backbone, 32k vocab (+25M embedding params),
    # 2048-token context ~ 9.5KB of text (information-matched to 8192 bytes)
    "small-bpe": dict(n_layer=12, n_head=12, n_embd=768, block_size=2048,
                      vocab_size=32768),
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (x * self.weight.float()).to(dtype)


def precompute_rope(head_dim: int, max_seq: int, theta: float):
    """Rotation frequencies for RoPE: positions are encoded by rotating
    q/k pairs of dims, so attention scores depend on *relative* offsets."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, inv_freq)  # (max_seq, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (B, n_head, T, head_dim)
    T = x.size(2)
    cos, sin = cos[:T].to(x.dtype), sin[:T].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.wqkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.wo = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.wqkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)


class SwiGLU(nn.Module):
    """Gated MLP: out = W2( silu(W_gate x) * (W_up x) ).
    Hidden dim chosen ~8/3 * d_model (rounded to 128) so total params match
    the classic 4x GELU MLP."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)
        hidden = 128 * round(hidden / 128)
        self.w_gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class ByteGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying

        cos, sin = precompute_rope(cfg.n_embd // cfg.n_head, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scale down residual-branch output projections (GPT-2 trick):
        # keeps the residual stream variance stable across depth
        for pn, p in self.named_parameters():
            if pn.endswith("wo.weight") or pn.endswith("w_down.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x, self.rope_cos, self.rope_sin)
        x = self.norm_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss
        # inference: only the last position is needed
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    def configure_optimizer(self, weight_decay, lr, betas, device_type):
        # decay only matrices; norms and embeddings stay undecayed
        decay = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        nodecay = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ]
        fused = device_type == "cuda"
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)

    @torch.no_grad()
    def generate(self, idx, max_new_bytes, temperature=1.0, top_k=None):
        for _ in range(max_new_bytes):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
