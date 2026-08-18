import torch
from model import GPT
from tokenizer_char import CharTokenizer

# Hyperparameters

device = "cuda" if torch.cuda.is_available() else "cpu"
block_size = 128  # how many characters of context the model sees at once
batch_size = 64   # how many sequences processed per training step
d_model = 128
n_heads = 4
n_layers = 4
dropout = 0.1
learning_rate = 3e-4
max_iters = 3000       # total number of training steps
eval_interval = 300    # how often we check train/val loss
eval_iters = 50        # how many batches to average over when checking loss


torch.manual_seed(42)
with open("data/shakespeare.txt","r") as f:
    text = f.read()

tok = CharTokenizer(text)
data = torch.tensor(tok.encode(text), dtype=torch.long)   # the ENTIRE dataset as one long list of ids , of type tensor

# Train and val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    """Sample `batch_size` random windows of length `block_size` from the data,
    plus their corresponding targets (same window, shifted by 1 character)."""
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))  # batch_size random start indices
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()   # disables gradient tracking. we're just measuring loss in this
def estimate_loss(model):
    """Average loss over several random batches -- a single batch's loss is
    noisy, averaging gives a more trustworthy read on how training is going."""
    model.eval()   # switches dropout off durintg infrence
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()   # switch dropout back on for continued training
    return out

def main():
    model = GPT(tok.vocab_size, d_model, n_heads, n_layers, block_size, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # using AdamW optimizer

    print(f"Training on device: {device}")
    for it in  range(max_iters+1):
        if it % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        x, y = get_batch("train")
        logits, loss = model(x,y)
        optimizer.zero_grad(set_to_none=True) # clear old gradients from the previous step
        loss.backward()  # compute new gradients for this step
        optimizer.step() # update every parameter using its gradient

    # save the trained weights + everything needed to reload the model later
    torch.save({
        "model_state": model.state_dict(),
        "config": dict(vocab_size=tok.vocab_size, d_model=d_model, n_heads=n_heads,
                        n_layers=n_layers, block_size=block_size, dropout=dropout),
        "stoi": tok.stoi, "itos": tok.itos,
    }, "checkpoint.pt")
    print("Saved checkpoint.pt")

if __name__ == "__main__":
    main()
