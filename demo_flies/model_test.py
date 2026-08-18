import math
import torch
from model import GPT
from tokenizer_char import CharTokenizer

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shakespeare.txt")
with open(data_path, "r") as f:
    text = f.read()
tok = CharTokenizer(text)

# small config
block_size = 16
model = GPT(
    vocab_size=tok.vocab_size,
    d_model=32,
    n_heads=4,
    n_layers=2,
    block_size=block_size,
)

# Tiny batch by hand
batch_size = 4
sample_text = text[:1000]    
ids = torch.tensor(tok.encode(sample_text))

# chop it into `batch_size` chunks of length `block_size`, and the corresponding "next character" targets (shifted by 1)
x = torch.stack([ids[i:i+block_size] for i in range(0, batch_size*block_size, block_size)])
y = torch.stack([ids[i+1:i+block_size+1] for i in range(0, batch_size*block_size, block_size)])

print("Input batch shape (B, T):", x.shape)   # (4, 16)
print("Target batch shape (B, T):", y.shape)  # (4, 16)

logits, loss = model(x, y) # calls forward function
print("\nLogits shape (B, T, vocab_size):", logits.shape)  # (4, 16, 65)
print("Loss:", loss.item())

expected_random_loss = math.log(tok.vocab_size)
print(f"Expected loss for a completely untrained/random model: ln({tok.vocab_size}) = {expected_random_loss:.4f}")

loss.backward() # Triggers Backprop, In PyTorch, the computational graph is built automatically, but backpropagation itself is executed on demand.
total_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
print(f"\nTotal gradient norm across all parameters: {total_grad_norm:.4f}")
print("(A nonzero number here means gradients successfully flowed through every layer.)")