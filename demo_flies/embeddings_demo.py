import torch
import torch.nn as nn

from tokenizer_char import CharTokenizer

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shakespeare.txt")
with open(data_path, "r") as f:
    text = f.read()
tok = CharTokenizer(text) # Tokenizer object

d_model = 8 # dimention of embedding vectos

# Currently Random matrix of shape (vocab size, d_model), nn.Embedding initializes them from a normal distribution automatically and also they are parameters as well, so gradient tracking is on
token_emb = nn.Embedding(tok.vocab_size, d_model)

print("Embedding table shape:", token_emb.weight.shape) # actual data is stoed in weightcle

char = "F" # Demo search
char_id = tok.stoi[char]
print(f"'{char} has id {char_id}")

id_tensor = torch.tensor([char_id]) # nn.Embeddings excepts a tensor
vector = token_emb(id_tensor) # Simply pick that row, that is its vector

print(f"Embedding vector for '{char}':\n{vector}")
print("Shape:", vector.shape)

sample = "Hi"
ids = torch.tensor(tok.encode(sample))
vectors = token_emb(ids)                   
print(f"\nEmbeddings for '{sample}':\n{vectors}")
print("Shape:", vectors.shape)

# Positional Embeddings - add order information, give each position a learnable position

block_size = 16 # the max sequence length our model will ever look at, which will also be number of rows in pos_emb

pos_emb = nn.Embedding(block_size, d_model) # d_model dim block_size vectors

sample = "Hi"
ids = torch.tensor(tok.encode(sample))
positions = torch.arange(len(sample))   
print("Position ids:", positions)

tok_vectors = token_emb(ids)          # tells "what character is this"
pos_vectors = pos_emb(positions)      # tells "where in the sequence is this"

x = tok_vectors + pos_vectors         # "which character" and "which position," blended into one vector of the same size. 
# Position and content are independent, learned separately, then fused by addition.

print("\nToken embeddings:\n", tok_vectors)
print("\nPosition embeddings:\n", pos_vectors)
print("\nCombined input (x = token_emb + pos_emb):\n", x)