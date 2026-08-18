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

d_model = 8    # size of each token's vector
d_k = 8        # size of the Query/Key/Value vectors for this single head

torch.manual_seed(42)

token_emb = nn.Embedding(tok.vocab_size, d_model)
pos_emb = nn.Embedding(16, d_model)

# Input x
sample = "Hi there"
ids = torch.tensor(tok.encode(sample))
positions = torch.arange(len(sample))
x = token_emb(ids) + pos_emb(positions) # # shape (T, d_model), T = 8 characters here

# Query (what am I looking for?), a Key (what do I contain, for others to match against?), and a Value (what do I actually hand over, if picked?). (d_model x d_k) matrix of learnable numbers.
W_q = nn.Linear(d_model, d_k, bias=False) # "Linear" in PyTorch means: output = input @ weight.T -- a matrix multiply. Basically a linear layer , a = s+y (no bias term as specified)
W_k = nn.Linear(d_model, d_k, bias=False)
W_v = nn.Linear(d_model, d_k, bias=False)

# Pass x to those linear layers
Q = W_q(x)   # (T, d_k) one Query vector per token
K = W_k(x)   # (T, d_k) one Key vector per token
V = W_v(x)   # (T, d_k) one Value vector per token

print("Q shape:", Q.shape)

# Attention Score: We compare every token's Query against every token's Key (dot product = similarity score).

scores = Q @ K.transpose(0,1)

scores = scores / math.sqrt(d_k) # Root d scaleing, so dot product doesn't grow. Without this softmax gives nearly one-hoy, giving near zero gradients almost everywhere.
# Dividing by sqrt(d_k) keeps the scores in a well-behaved range regardless of vector size.

print("\nRaw attention scores (T x T):\n", scores)

# causal mask: token i must not see tokens AFTER position i, used when we are buidling a decoder like GPT
T = x.shape[0]
mask = torch.tril(torch.ones(T, T))   # lower-triangular matrix of 1s and 0s
print("\nCausal mask (1 = allowed, 0 = blocked):\n", mask)

scores = scores.masked_fill(mask == 0, float("-inf")) # wherever mask is 0 (future positions), set the score to -infinity. After softmax, -infinity becomes a weight of exactly 0.
# Turn those scores into a probability distribution (softmax) — "how much weight to put on each other position." per row

attn_weights = torch.softmax(scores, dim=-1)
print("\nAttention weights after softmax (each row sums to 1):\n", attn_weights)
print("\nRow sums (should all be 1.0):", attn_weights.sum(dim=-1))

# Use those weights to take a weighted average of everyone's Value vectors. That weighted average is the output for this position.

out = attn_weights @ V   # (T, T) @ (T, d_k) -> (T, d_k)
print("\nOutput of attention (T, d_k):\n", out)
print("Shape:", out.shape)

# Multi-Head Attention

n_heads = 4 # number of heads
d_k_head = d_model // n_heads   # 8 // 4 = 2 , we later concatenate these 4 to get final matrices, helps to learn multiple patters

# In practice we don't create separate W_q/W_k/W_v per head -- we make ONE, linear layer that outputs all heads' worth of Q (or K, or V) at once,
# then reshape/split it into heads. This is just a speed optimization, conceptually it's identical to n_heads separate small attention blocks.

W_q_mh = nn.Linear(d_model, d_model, bias=False)
W_k_mh = nn.Linear(d_model, d_model, bias=False)
W_v_mh = nn.Linear(d_model, d_model, bias=False)
W_out  = nn.Linear(d_model, d_model, bias=False)  # final mixing layer

T = x.shape[0]  # 8, number of characters

Q = W_q_mh(x)   # (T, d_model) = (8, 8)
K = W_k_mh(x)
V = W_v_mh(x)

# reshape (T, d_model) -> (T, n_heads, d_k_head) -> (n_heads, T, d_k_head), so that each head's slice is a separate (T, d_k_head) matrix we can run attention on independently and parallely
Q = Q.view(T, n_heads, d_k_head).transpose(0, 1)  # (n_heads, T, d_k_head)
K = K.view(T, n_heads, d_k_head).transpose(0, 1)
V = V.view(T, n_heads, d_k_head).transpose(0, 1)

print("Q reshaped for multi-head:", Q.shape)

# Attention , on multiple heads, PyTorch's @ operator batches over any leading dimensions (here: n_heads).

scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k_head)   # (n_heads, T, T)
mask = torch.tril(torch.ones(T, T))
scores = scores.masked_fill(mask == 0, float("-inf"))
attn_weights = torch.softmax(scores, dim=-1)              # (n_heads, T, T)

head_outputs = attn_weights @ V   # (n_heads, T, d_k_head)
print("Per-head outputs shape:", head_outputs.shape)

# merge heads back together: (n_heads, T, d_k_head) -> (T, n_heads, d_k_head) -> (T, d_model)
merged = head_outputs.transpose(0, 1).contiguous().view(T, d_model)
print("Merged heads shape:", merged.shape)

final_out = W_out(merged)   # Use final layer to merge the output, W_out allows the model to learn combinations such as: output_1 = 0.2a + 0.5c - 0.1e + ...
print("\nFinal multi-head attention output:\n", final_out)
print("Shape:", final_out.shape)