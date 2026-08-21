import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttentionFlash(nn.Module):
    """Identical structure to CausalSelfAttention in model.py, but the actual
    attention computation (scores, mask, softmax, weighted sum) is replaced
    by ONE call to PyTorch's fused Flash Attention kernel."""
    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.resid_dropout = nn.Dropout(dropout)
        # no manual causal mask buffer needed anymore -- SDPA builds
        # the causal mask internally when we pass is_causal=True.

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)   # (B, n_heads, T, d_k)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Replaces: scores = q@k.T/sqrt(d_k), mask, softmax, attn@v
        # It handles scaling by 1/sqrt(d_k) internally by default.
        out = F.scaled_dot_product_attention( # automatically dispatches to a Flash Attention implementation (or a couple of other optimized kernels) depending on your GPU and input shapes
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,   # only apply dropout during training
            is_causal=True,   # tells the kernel to apply a causal mask internally, without us building one
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
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


class BlockFlash(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttentionFlash(d_model, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTFlash(nn.Module):
    """Identical to GPT in model.py, just built from BlockFlash instead of Block."""
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 block_size=128, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            BlockFlash(d_model, n_heads, block_size, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPTFlash initialized: {n_params/1e6:.2f}M parameters")

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)

        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss