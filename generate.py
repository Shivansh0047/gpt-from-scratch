import torch
from model import GPT
from model_kv import GPTKV
from tokenizer_bpe import BPETokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# load everything we saved during training
ckpt = torch.load("checkpoint_tinystories.pt", map_location=device, weights_only=False)
cfg = ckpt["config"]
# stoi = ckpt["stoi"]
# itos = ckpt["itos"]

tok = BPETokenizer()

# model = GPT(**cfg).to(device)             # rebuild the model with the SAME architecture
# model.load_state_dict(ckpt["model_state"])  # load the trained weights into it
# model.eval()       # Put the model in eval mode # turn off dropout for generation

model = GPTKV(**cfg).to(device)             # rebuild the model with the SAME architecture, KV-cached version
model.load_state_dict(ckpt["model_state"])  # load the trained weights into it -- same state_dict keys work for both
model.eval()       # Put the model in eval mode # turn off dropout for generation
'''
def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join(itos[i] for i in ids)
'''

@torch.no_grad() 
def generate(model, idx, max_new_tokens, temperature=0.8, top_k=40):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]        # only keep the last block_size tokens as context 
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            logits, _ = model(idx_cond)     # # forward pass, no targets -> loss is None in 16 bit float
        logits = logits[:, -1, :]                     # we only care about the prediction for the NEXT token
        logits = logits / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k) # Take only top k highest tokens, an alternative is top-p (Dynamically selects the smallest set of top tokens whose cumulative probability exceeds threshold p (e.g., \p = 0.90))
            logits[logits < v[:, [-1]]] = float("-inf")  # zero out everything except the top_k choices

        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)  # SAMPLE (not argmax) from the distribution. we Renormalization & Random Draw from the top k values
        idx = torch.cat([idx, idx_next], dim=1)              # append and repeat
    return idx

# prompt = "ROMEO:"
prompt = "Once upon a time"
context = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)  # shape (1, len(prompt))

# out = generate(model, context, max_new_tokens=500)   # old, non-cached generation loop (needs GPT, not GPTKV)

assert context.shape[1] + 100 <= model.block_size, "prompt + max_new_tokens must fit inside block_size"
out = model.generate(context, max_new_tokens=100)   # new: KV-cached generation, built into GPTKV itself

print(tok.decode(out[0].tolist()))