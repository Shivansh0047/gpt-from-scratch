import torch
import torch.nn as nn

from tokenizer_char import CharTokenizer

with open("data/shakespeare.txt", "r") as f:
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