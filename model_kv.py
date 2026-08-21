import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttentionKV(nn.Module):
    """Same math as CausalSelfAttention in model.py, but forward() can accept
    and return a KV cache.
    """

    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        assert d_model %n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model) # nstead of defining three separate linear layers (self.q_proj, self.k_proj, and self.v_proj), we combine them into a single fused matrix multiplication for performance.
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x ,cache=None):
        B, T, C = x.shape   # T here is however many NEW tokens we're processing this call

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)   # (B, n_heads, T, d_k)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        if cache is not None: # Passed during the initial prompt step (prefill).
            cache_k, cache_v = cache
            if cache_k is not None:
                k = torch.cat([cache_k, k], dim=2)   # append new K onto cached K, along the sequence dim
                v = torch.cat([cache_v, v], dim=2)
            new_cache = (k, v)   # this now holds EVERYTHING seen so far (old + new)
        else:
            new_cache = None # standared forward pass

        L = k.shape[2]   # total sequence length now available in K/V (old + new)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)   # (B, n_heads, T, L)
        if T > 1: # prefill: x has T > 1 tokens (e.g. the initial prompt), cache starts empty,  we DO need the causal mask (multiple queries attending to agrowing set of keys, must not see each other's futures).
            causal_mask = self.mask[:, :, L - T:L, :L]
            scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        # x has T == 1 token (one new token during generation), cache already holds everything before it. No mask needed
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out, new_cache


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

class BlockKV(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttentionKV(d_model, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x, cache=None):
        attn_out, new_cache = self.attn(self.ln1(x), cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_cache

class GPTKV(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,block_size=128, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.n_layers = n_layers
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            BlockKV(d_model, n_heads, block_size, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPTKV initialized: {n_params/1e6:.2f}M parameters")

    def forward(self, idx, cache=None, start_pos=0):
        """
        idx: (B, T) -- T new tokens to process this call (T can be > 1 for prefill,
             or exactly 1 for a single decode step).
        cache: either None (no caching -- behaves like the original GPT) or a list
             of (k, v) tuples, one per layer, or a list of Nones for a fresh cache.
        start_pos: the position INDEX of the first token in `idx`. Needed because,
             in decode mode, idx is just the one newest token -- we still need to
             tell pos_emb.
        """
        B, T = idx.shape
        positions = torch.arange(start_pos, start_pos + T, device=idx.device)

        x = self.token_emb(idx) + self.pos_emb(positions)   # (B, T, d_model)
        x = self.drop(x)

        new_caches = []
        for i, block in enumerate(self.blocks):
            layer_cache = cache[i] if cache is not None else None
            x, new_cache = block(x, layer_cache)
            new_caches.append(new_cache)

        x = self.ln_f(x)
        logits = self.head(x)   # (B, T, vocab_size)
        return logits, new_caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive generation USING the cache.
        """
        B, T_prompt = idx.shape

        # positions used will range from 0 to (T_prompt + max_new_tokens - 1) --
        # pos_emb only has `block_size` valid rows, so this MUST fit.
        assert T_prompt + max_new_tokens <= self.block_size, (
            f"prompt length ({T_prompt}) + max_new_tokens ({max_new_tokens}) = "
            f"{T_prompt + max_new_tokens} exceeds block_size ({self.block_size}). "
            f"This simple KV-cache implementation doesn't support generation "
            f"beyond the trained context window -- reduce max_new_tokens."
        )

        # prefill: process the entire prompt at once
        cache = [(None, None)] * self.n_layers   # start with an empty cache tuple per layer, Each of the n_layers blocks has its own K/V cache (a block's attention only ever needs its own layer's cached K/V, never another layer's)
        logits, cache = self(idx, cache=cache, start_pos=0)

        # we only need the LAST position's logits to pick the first new token
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

        # decode loop: one new token at a time, cache does the heavy lifting
        for step in range(max_new_tokens - 1):
            current_pos = T_prompt + step   # position of the token we're about to feed in
            logits, cache = self(next_id, cache=cache, start_pos=current_pos)

            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

        return idx