import torch
import torch.nn as nn
import math

torch.manual_seed(0)

d_model = 8
n_heads = 2
d_k = d_model // n_heads
vocab_size = 20
block_size = 16

token_emb = nn.Embedding(vocab_size, d_model)
pos_emb = nn.Embedding(block_size, d_model)

# The Q/K/V projection matrices
W_q = nn.Linear(d_model, d_model, bias=False)
W_k = nn.Linear(d_model, d_model, bias=False)
W_v = nn.Linear(d_model, d_model, bias=False)

# Pretend token sequence
full_sequence = torch.tensor([5, 12, 3, 7, 9]) 

# METHOD 1: naive -- recompute K/V for the ENTIRE sequence so far, every step
print("=== Naive: recompute everything each step ===")
naive_K_at_each_step = []   # we'll save the full K tensor computed at each step, to compare later

for step in range(1, len(full_sequence) + 1):
    ids_so_far = full_sequence[:step]                  # e.g. step=3 -> tokens [5, 12, 3]
    positions = torch.arange(step)
    x = token_emb(ids_so_far) + pos_emb(positions)       # (step, d_model) -- recomputed from scratch

    K = W_k(x)   # (step, d_model) -- K for EVERY token, recomputed from scratch, even old ones
    naive_K_at_each_step.append(K)
    print(f"step {step}: recomputed K for all {step} token(s), shape {K.shape}")

# METHOD 2: cached -- compute K/V for the NEW token only, append to a growing cache

print("\n=== Cached: compute only the new token, reuse the rest ===")
cache_K = None   # will grow to hold every token's K, one row at a time
cache_V = None

cached_K_at_each_step = []  # save the full accumulated cache at each step, for comparison

for step in range(1, len(full_sequence) + 1):
    new_token_id = full_sequence[step - 1:step]   # JUST the newest token, e.g. tensor([3]) at step 3
    position = torch.tensor([step - 1])            # its actual position index (0-based)

    x_new = token_emb(new_token_id) + pos_emb(position)   # (1, d_model) -- ONLY this one token

    K_new = W_k(x_new)   # (1, d_model) -- K computed ONLY for the new token
    V_new = W_v(x_new)

    if cache_K is None:
        cache_K = K_new
        cache_V = V_new
    else:
        cache_K = torch.cat([cache_K, K_new], dim=0)   # append the new row onto the growing cache
        cache_V = torch.cat([cache_V, V_new], dim=0)

    cached_K_at_each_step.append(cache_K.clone())
    print(f"step {step}: computed K for 1 new token, cache now holds {cache_K.shape[0]} token(s) total")

print("\n=== Verification ===")
for step in range(len(full_sequence)):
    naive = naive_K_at_each_step[step]
    cached = cached_K_at_each_step[step]
    match = torch.allclose(naive, cached, atol=1e-6)
    print(f"step {step+1}: naive and cached K match? {match}")

# Checkig with attention
print("\n\n=== Full attention, naive vs cached, step by step ===")

def multi_head_attention_full(x, W_q, W_k, W_v):
    """Naive: given ALL tokens so far, compute Q/K/V for all of them,
    then run standard causal attention. This is what plain generate() does."""
    T = x.shape[0]
    Q = W_q(x).view(T, n_heads, d_k).transpose(0, 1)   # (n_heads, T, d_k)
    K = W_k(x).view(T, n_heads, d_k).transpose(0, 1)
    V = W_v(x).view(T, n_heads, d_k).transpose(0, 1)

    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)   # (n_heads, T, T)
    mask = torch.tril(torch.ones(T, T))
    scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = (attn @ V).transpose(0, 1).contiguous().view(T, d_model)
    return out   # (T, d_model) -- output for EVERY position, we only actually need the LAST row

def attention_step_cached(x_new, position, W_q, W_k, W_v, cache_K, cache_V):
    """Cached: given only the NEW token, compute its Q, compute its K/V and
    append to cache, then attend this ONE query against the FULL cache
    (old + new). No mask needed"""
    Q_new = W_q(x_new).view(1, n_heads, d_k).transpose(0, 1)   # (n_heads, 1, d_k)
    K_new = W_k(x_new).view(1, n_heads, d_k).transpose(0, 1)
    V_new = W_v(x_new).view(1, n_heads, d_k).transpose(0, 1)

    if cache_K is None:
        cache_K, cache_V = K_new, V_new
    else:
        cache_K = torch.cat([cache_K, K_new], dim=1)   # concat along the sequence dim
        cache_V = torch.cat([cache_V, V_new], dim=1)

    # attention: 1 query vs. ALL cached keys/values (no mask needed, see above)
    scores = (Q_new @ cache_K.transpose(-2, -1)) / math.sqrt(d_k)   # (n_heads, 1, L)
    attn = torch.softmax(scores, dim=-1)
    out = (attn @ cache_V)   # (n_heads, 1, d_k)
    out = out.transpose(0, 1).contiguous().view(1, d_model)          # (1, d_model)
    return out, cache_K, cache_V

cache_K, cache_V = None, None

for step in range(1, len(full_sequence) + 1):
    ids_so_far = full_sequence[:step]
    positions_so_far = torch.arange(step)
    x_full = token_emb(ids_so_far) + pos_emb(positions_so_far)

    # naive: recompute attention over everything, take the LAST position's output
    naive_out_all = multi_head_attention_full(x_full, W_q, W_k, W_v)
    naive_last = naive_out_all[-1:, :]   # (1, d_model) -- just the newest token's output

    # cached: process only the new token 
    x_new = x_full[-1:, :]                          # (1, d_model) -- just the new token's embedding
    cached_out, cache_K, cache_V = attention_step_cached(
        x_new, step - 1, W_q, W_k, W_v, cache_K, cache_V
    )

    match = torch.allclose(naive_last, cached_out, atol=1e-5)
    print(f"step {step}: naive output vs cached output match? {match}")