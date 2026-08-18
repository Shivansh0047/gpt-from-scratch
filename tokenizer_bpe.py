import tiktoken

# gpt2 encoding: the exact tokenizer GPT-2 was trained with, vocab size 50257.
# This contains the rules of which wrds to merge together. How BPE works is it finds the most frequent adjacent words
# and it merge them together, builds a table, and repeats this process

enc = tiktoken.get_encoding("gpt2")

class BPETokenizer:
    def __init__(self):
        self.vocab_size = enc.n_vocab

    def encode(self, s: str):
        return enc.encode(s, allowed_special={"<|endoftext|>"}) # allowed_special: by default tiktoken treats "<|endoftext|>" as a literal
        # string to tokenize character-by-character. We explicitly allow it <|endoftext|> to be tokenized as its one special control token
        # rather than being chopped up into 7 ordinary text tokens

    def decode(self, ids):
        return enc.decode(ids)


if __name__ == "__main__":
    tok = BPETokenizer()
    print(f"Vocab size: {tok.vocab_size}")

    sample = "Once upon a time, there was a little girl."
    ids = tok.encode(sample)
    print(f"\n'{sample}'")
    print(f"-> {len(ids)} tokens: {ids}")
    print(f"decoded back -> '{tok.decode(ids)}'")

    print("\nToken-by-token breakdown:")
    for token_id in ids:
        piece = tok.decode([token_id])
        print(f"  {token_id:6d} -> {piece!r}")