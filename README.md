# GPT from Scratch

A decoder-only Transformer, implemented from first principles in PyTorch — no
Hugging Face, no pre-built attention layers. Built in two phases: a character-level
model trained on Shakespeare, then a BPE-tokenized model trained on TinyStories.

## What's in here

**Core implementation**
- `model.py` — the GPT itself: causal multi-head self-attention, feed-forward
  blocks, residual connections, LayerNorm, stacked into a full Transformer.
- `tokenizer_char.py` — character-level tokenizer (Phase 1a).
- `tokenizer_bpe.py` — BPE tokenizer via `tiktoken` (GPT-2 vocabulary, Phase 1b).
- `train.py` — training loop: batching, AdamW, mixed-precision (`autocast` +
  `GradScaler`), checkpointing.
- `generate.py` — autoregressive sampling from a trained checkpoint, with
  temperature and top-k.

**`demo_files/` — step-by-step build-up (kept for reference — these are how the
model was actually developed, one mechanism at a time, before being assembled
into `model.py`)**
- `demo_files/embeddings_demo.py` — token + positional embeddings in isolation.
- `demo_files/attention_demo.py` — single-head, then multi-head, scaled
  dot-product attention with a causal mask, worked through by hand.
- `demo_files/block_demo.py` — one full Transformer block (attention + FFN +
  residuals + LayerNorm) assembled and shape-checked.
- `demo_files/model_test.py` — sanity-checks the assembled `GPT` class: correct
  output shapes, and confirms untrained loss lands near `ln(vocab_size)` as
  expected for a random baseline.

## Phase 1a — character-level, Tiny Shakespeare

65-character vocabulary, ~0.83M parameters, trained 3000 steps (~a few minutes
on an RTX 4050).

Loss: **4.31 → 1.58** (train), **4.30 → 1.76** (val).

Sample output (prompt: `"ROMEO:"`):
```
ROMEO:
He Clain, but I'll so, stain reson the day.
DUKE VOLYCENTIO:
To hath I dreases, when a clanded the lord friend,
The joy; facends and new mard honour and some,
```

Correct play-script formatting and character-name patterns, real English words
mixed in, but no real semantic coherence — expected at this scale and from a
character-level vocabulary, which forces the model to spend its capacity
re-deriving spelling rather than meaning.

## Phase 1b — BPE tokenization, TinyStories

Switched to GPT-2's BPE vocabulary (50,257 tokens, via `tiktoken`) and TinyStories
as the dataset. Scaled the model up (~28.9M parameters) and trained 8000 steps
(~15 min on an RTX 4050, using mixed-precision training).

Loss: **10.98 → 2.32** (train), **10.98 → 2.43** (val). (Not directly comparable
to Phase 1a's numbers — a 50k-token vocabulary is a fundamentally harder
prediction task than a 65-character one.)

Sample output (prompt: `"Once upon a time"`):

```
Once upon a time, there was a boy named Tim. Tim was a good boy who liked to
play with his toy cars. He would make sand with his toy cars was all different
things, and Tim had a lot of fun together.
One day, Tim saw a big, scary dog. The dog was running and ran away. Tim wanted
to play with the wheel, but he didn't want to stop. He was very sad and asked
his friend, "Do not touch the dog, Max?" Max said, "I lost the car. Maybe it is
too heavy."
Tim and the dog looked at each other and said, "I am sorry, Max because I
should not lose it." They played together and had fun. They became good friends.
<|endoftext|>
```

A clear jump over Phase 1a: consistent grammar, working dialogue with correct
punctuation, consistent character names within a story, and the model
independently learned to emit `<|endoftext|>` to end one story and start the
next — a behavior it picked up purely from the training data's formatting,
never hard-coded. Plot logic still wanders at times (objects and actions
occasionally don't quite make sense), which tracks with a val loss of ~2.4 and
a 29M-parameter model — the ceiling here is architecture/scale/training time,
not a bug.

## Phase 2 — KV Caching

Added a parallel implementation (`model_kv.py`) of the same architecture, where
each attention layer can accept and extend a cache of previously-computed
Key/Value tensors instead of recomputing them from scratch every generation step.

**Correctness check:** loaded the Phase 1b trained weights into both `GPT`
(`model.py`) and `GPTKV` (`model_kv.py`), ran the same prompt through both.
Logits matched exactly (max absolute difference: `0.0`), confirming the cached
version is a pure speed optimization — mathematically identical computation,
restructured to avoid redundant work.

**Speed:** generating 100 tokens, KV-cached generation ran consistently faster
than the uncached version (~1.5-2x in testing), with the gap expected to widen
further at longer generation lengths, since uncached generation costs `O(N³)`
total work as sequence length `N` grows (recomputing full attention over an
ever-growing sequence, every step) versus `O(N²)` for the cached version
(computing attention for only the new token each step, against a cache that's
been paid for once).

**Two real bugs hit and fixed along the way, worth noting since they're
representative of the kind of mistakes this technique is prone to:**
1. **Position embedding overflow** — generating beyond `block_size` total
   tokens (prompt + generated) asks the positional embedding table for a row
   index that doesn't exist. Fixed with an explicit assertion before
   generation starts, rather than letting it fail as a confusing async CUDA
   error mid-run. (This limitation exists in the original uncached model too,
   in principle — it's just silently masked there by unconditional sequence
   cropping.)
2. **Silent cache-drop bug** — initializing each layer's cache slot as bare
   `None` (instead of `(None, None)`) caused the attention layer's `cache is
   not None` check to fail every time, so computed K/V were silently never
   stored. The model appeared to run without error, but generated fluent,
   grammatically fine, *completely incoherent* text — because it had zero
   memory of any token beyond the current one. A good reminder that a
   caching bug won't always crash; it can silently degrade output quality
   instead, which is arguably worse.

**Limitation, not yet solved:** like the original model, this implementation
still can't generate beyond `block_size` total tokens — real long-context
inference needs either a sliding-window cache (evict oldest entries) or an
architecture change (e.g. rotary position embeddings) that doesn't hard-code
a maximum sequence length.

## Run it

```bash
pip install torch tiktoken

# Phase 1a
python train.py          # (with tokenizer_char.py config active)
python generate.py

# Phase 1b
python train.py          # (with tokenizer_bpe.py config active)
python generate.py       # loads checkpoint_tinystories.pt
```
`preflight_check.py` is worth running first when changing model size or
hardware — it catches GPU out-of-memory issues in seconds rather than minutes
into a real training run, and gives a steady-state time-per-step estimate
(discarding the first step, which pays a one-time CUDA warmup cost).

The `demo_files/` scripts can be run directly, from either the repo root or
from inside `demo_files/` itself — e.g. `python demo_files/attention_demo.py`.

## What's next

- **Flash Attention** — swapping the manual attention implementation for a
  fused kernel, benchmarked against the from-scratch version.