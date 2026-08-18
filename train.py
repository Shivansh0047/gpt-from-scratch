import torch
from model import GPT
from tokenizer_char import CharTokenizer
from tokenizer_bpe import BPETokenizer   

# Hyperparameters

device = "cuda" if torch.cuda.is_available() else "cpu"
block_size = 128  # how many characters of context the model sees at once
batch_size = 16   # how many sequences processed per training step
d_model = 256 # d_model is the size of the vector each token gets represented by as it flows through the network, the parameter count grows by d_model², as most weight matrices are d_model × d_model or d_model × 4*d_model
n_heads = 8
n_layers = 4
dropout = 0.1
learning_rate = 3e-4
max_iters = 8000       # total number of training steps
eval_interval = 500    # how often we check train/val loss
eval_iters = 20        # how many batches to average over when checking loss


torch.manual_seed(42)
with open("data/tinystories.txt", "r", encoding="utf-8") as f:
    text = f.read()

# tok = CharTokenizer(text)
tok = BPETokenizer() # use BPETokenizer now
data = torch.tensor(tok.encode(text), dtype=torch.long)   # the ENTIRE dataset as one long list of ids , of type tensor

# Train and val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
# GradScaler: manages the loss-scaling trick so in 16 bit float, gradient are not rounded to zero enabled=False automatically on CPU, where mixed precision doesn't apply.

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
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")): # autocast is what actually makes the forward pass run parts of the model in float16
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
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            logits, loss = model(x, y) # forward pass (in float16 where safe)
        optimizer.zero_grad(set_to_none=True) # clear old gradients from the previous step
        scaler.scale(loss).backward()            # compute new gradients for this step, scale loss up, then compute gradients (protects small values in fp16)
        scaler.step(optimizer)                    # unscales gradients, skips the step if any overflowed, calls optimizer.step() internally to update every parameter using its gradient
        scaler.update()                            # adjusts the scale factor for next iteration

    # save the trained weights + everything needed to reload the model later
    '''
    torch.save({
        "model_state": model.state_dict(),
        "config": dict(vocab_size=tok.vocab_size, d_model=d_model, n_heads=n_heads,
                        n_layers=n_layers, block_size=block_size, dropout=dropout),
        "stoi": tok.stoi, "itos": tok.itos,
    }, "checkpoint_tinyShekespear.pt")
    print("Saved checkpoint_tinyShekespear.pt")
    '''

    torch.save({
        "model_state": model.state_dict(),
        "config": dict(vocab_size=tok.vocab_size, d_model=d_model, n_heads=n_heads,
                        n_layers=n_layers, block_size=block_size, dropout=dropout),
    }, "checkpoint_tinystories.pt")
    print("Saved checkpoint_tinystories.pt")

if __name__ == "__main__":
    main()
