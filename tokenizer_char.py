class CharTokenizer: # Tokenize at char level
    def __init__(self,text: str):
        chars = sorted(list(set(text))) # unique char
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)} # stirng to int
        self.itos = {i: ch for i, ch in enumerate(chars)} # int to string

    def encode(self, s:str):
        return [self.stoi[c] for c in s] # returns list of numbers

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids) # returns sting
 

if __name__ == "__main__":
    with open("data/shakespeare.txt", "r") as f:
        text = f.read()

        tok = CharTokenizer(text)

        print(f"Dataset length: {len(text)} Characters")
        print(f"Vocab size: {tok.vocab_size}")

        sample = "First Citizen:"
        ids = tok.encode(sample)
        print(f"'{sample}' -> {ids}")
        print(f"decoded back -> '{tok.decode(ids)}'")
