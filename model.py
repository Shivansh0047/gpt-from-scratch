import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    """Same multi-head attention math from block_demo.py, packaged as a
    reusable nn.Module class instead of loose variables + a function."""
    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)   # produces Q,K,V in one matmul
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))
        # register_buffer: stores this tensor as part of the module, moves with .to(device), gets saved in checkpoints) but it's NOT a learnable

    def forward(self, x):
        B, T, C = x.shape   # B = batch size (how many sequences at once), T = sequence length, C = d_model; 
        # Real training processes many sequences at once for efficiency — shape (B, T, d_model). All the same matrix operations just gain one more leading dimension;
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """One full transformer layer: attention (+residual) then FFN (+residual)."""
    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 block_size=128, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)

        # n_layers separate Block instances, each with its own, independently-learned weights, applied one after another.
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, block_size, dropout) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)          # one final norm after the last block
        self.head = nn.Linear(d_model, vocab_size, bias=False)  # d_model -> vocab_size logits

        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT initialized: {n_params/1e6:.2f}M parameters")

    def forward(self, idx, targets=None):
        # idx: (B, T) integer token ids
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)

        x = self.token_emb(idx) + self.pos_emb(pos)   # (B, T, d_model)
        x = self.drop(x)

        for block in self.blocks:      # run through every stacked block in order
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)          # (B, T, vocab_size) -- a score per possible next-character, per position

        loss = None
        if targets is not None:
            # compare predicted logits to the actual next character at every position
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
