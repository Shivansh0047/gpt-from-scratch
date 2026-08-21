# GPT from Scratch

A decoder-only Transformer, implemented from first principles in PyTorch — no
Hugging Face, no pre-built attention layers. Built in two phases: a character-level
model trained on Shakespeare, then a BPE-tokenized model trained on TinyStories.

## What's in here

**Core implementation**
- `model.py` — the GPT itself: causal multi-head self-attention, feed-forward
  blocks, residual connections, LayerNorm, stacked into a full Transformer.
- `model_kv.py` — KV-cached version of the same architecture for efficient
  autoregressive inference.
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

**Benchmark:** generating 100 tokens from the same prompt:

| Implementation | Time | Speedup |
|---|---:|---:|
| Plain (no cache) | 1.31 s | 1.00× |
| KV-cached | 0.90 s | **1.46×** |

The KV-cached version was **1.46× faster** in this benchmark. The advantage comes
from avoiding repeated computation of Key/Value tensors for tokens that have
already been processed.

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
   `None` (instead of `(None, None)`) caused the attention layer's
   `cache is not None` check to fail every time, so computed K/V were silently
   never stored. The model appeared to run without error, but generated fluent,
   grammatically fine, *completely incoherent* text — because it had zero
   memory of any token beyond the current one. A good reminder that a caching
   bug won't always crash; it can silently degrade output quality instead,
   which is arguably worse.

**Limitation, not yet solved:** like the original model, this implementation
still can't generate beyond `block_size` total tokens — real long-context
inference needs either a sliding-window cache (evict oldest entries) or an
architecture change (e.g. rotary position embeddings) that doesn't hard-code
a maximum sequence length.

## Phase 3 — Flash Attention

Added a Flash Attention implementation (`GPTFlash`) using PyTorch's
scaled-dot-product attention (SDPA). The implementation uses the same trained
weights as the original model and replaces the manually materialized attention
computation with PyTorch's optimized attention kernel.

**Correctness check:** loaded the same Phase 1b checkpoint into both the plain
and Flash Attention models. The logits matched within floating-point precision:

- Logits match: **True**
- Maximum absolute difference: **0.00000477**

The tiny difference is expected from floating-point rounding and the different
computation order used by the optimized kernel.

**Initial benchmark at `T = 128`:**

| Implementation | Time/step | Peak memory |
|---|---:|---:|
| Plain (manual attention) | 109.03 ms | 2.422 GB |
| Flash Attention (SDPA) | 108.04 ms | 2.505 GB |

At this short sequence length, the performance difference is negligible.
Flash Attention's advantage becomes much more visible as the sequence length
increases.

**Scaling benchmark — forward pass only:**

| Sequence length (`T`) | Plain (ms) | Flash (ms) | Speedup | Plain memory | Flash memory |
|---:|---:|---:|---:|---:|---:|
| 128 | 7.11 | 7.31 | 0.97× | 0.939 GB | 0.939 GB |
| 256 | 13.86 | 10.74 | **1.29×** | 1.043 GB | 1.043 GB |
| 512 | 34.80 | 25.66 | **1.36×** | 1.254 GB | 1.254 GB |
| 1024 | 97.18 | 52.14 | **1.86×** | 1.682 GB | 1.682 GB |

The benchmark shows the expected trend: Flash Attention becomes increasingly
effective as sequence length grows. At `T = 1024`, the optimized implementation
is **1.86× faster** than the manually implemented attention.

The memory figures above measure total peak GPU memory allocated by the model,
rather than the attention matrix alone, so the difference in attention-specific
memory usage is not directly visible in this benchmark.

The key benefit demonstrated here is improved attention runtime at longer
sequence lengths while preserving the exact model computation.

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

The `demo_files/` scripts can be run directly, from either the repo root or
from inside `demo_files/` itself — e.g. `python demo_files/attention_demo.py`.