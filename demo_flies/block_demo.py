import torch
import torch.nn as nn
import math
from tokenizer_char import CharTokenizer

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shakespeare.txt")
with open(data_path, "r") as f:
    text = f.read()
tok = CharTokenizer(text)

d_model = 8
n_heads = 4
d_k_head = d_model // n_heads
torch.manual_seed(0)

token_emb = nn.Embedding(tok.vocab_size, d_model)
pos_emb = nn.Embedding(16, d_model)

sample = "Hi there"
ids = torch.tensor(tok.encode(sample))
positions = torch.arange(len(sample))
x = token_emb(ids) + pos_emb(positions)   # (T, d_model) = (8, 8)
T = x.shape[0]

# Multi-Head Attention wrapped as a reusable function

W_q = nn.Linear(d_model, d_model, bias=False)
W_k = nn.Linear(d_model, d_model, bias=False)
W_v = nn.Linear(d_model, d_model, bias=False)
W_out = nn.Linear(d_model, d_model, bias=False)
mask = torch.tril(torch.ones(T, T))

def multi_head_attention(x):
    Q = W_q(x).view(T, n_heads, d_k_head).transpose(0, 1)
    K = W_k(x).view(T, n_heads, d_k_head).transpose(0, 1)
    V = W_v(x).view(T, n_heads, d_k_head).transpose(0, 1)
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k_head)
    scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = (attn @ V).transpose(0, 1).contiguous().view(T, d_model)
    return W_out(out)

# Feed Forward Network - applied to each position independently (no mixing across positions here — attention already did that, gives ability to do complex reasoning

ffn = nn.Sequential(
    nn.Linear(d_model, 4 * d_model),  # expand: 8 -> 32
    nn.GELU(),                       
    nn.Linear(4 * d_model, d_model),  # project back down: 32 -> 8
)

# LayerNorm
ln1 = nn.LayerNorm(d_model)   # applied before attention
ln2 = nn.LayerNorm(d_model)   # applied before the FFN

print("Input x:\n", x)

# One full Transformer Block

# step 1: attention with residual block (helps to flow gradient easily, preventing gradient vanishing)
attn_out = multi_head_attention(ln1(x))   # normalize FIRST, then attend ("pre-LN")
x = x + attn_out   # residual: add attention's output to the original x, so the residual "highway" for gradients stays completely unobstructed — norm only affects what attention sees, not what gets carried forward.
print("\nAfter attention + residual:\n", x)

# Sub-step 2: feed-forward with residual
ffn_out = ffn(ln2(x))
x = x + ffn_out
print("\nAfter FFN + residual (this is the Block's final output):\n", x)
print("\nShape (should match input shape):", x.shape)