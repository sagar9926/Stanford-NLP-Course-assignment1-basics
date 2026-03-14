"""Script to load and test the BPE tokenizer with TinyStories data."""
import json
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer

# Get the data directory relative to this file
DATA_DIR = Path(__file__).parent.parent / "data"

# Load and convert merges
with open(DATA_DIR / "ts_merges.json", "r") as f:
    raw_merges = json.load(f)
merges = [(bytes(pair[0]), bytes(pair[1])) for pair in raw_merges]

# Load and convert vocab
with open(DATA_DIR / "ts_vocab.json", "r") as f:
    raw_vocab = json.load(f)
vocab = {int(k): bytes(v) for k, v in raw_vocab.items()}

# Create tokenizer
tokenizer = Tokenizer(vocab, merges)

if __name__ == "__main__":
    # Test encoding
    test_text = "Hello, world!"
    print(f"Text: {test_text!r}")
    encoded = tokenizer.encode(test_text)
    print(f"Encoded: {encoded}")
