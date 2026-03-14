"""Test script for the BPE tokenizer encode function."""
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_basics.tokenizer import Tokenizer

# Get the data directory
DATA_DIR = Path(__file__).parent.parent / "data"

# Load and convert merges
print("Loading merges...")
with open(DATA_DIR / "ts_merges.json", "r") as f:
    raw_merges = json.load(f)
merges = [(bytes(pair[0]), bytes(pair[1])) for pair in raw_merges]
print(f"Loaded {len(merges)} merges")

# Print sample merges
print("\n" + "=" * 50)
print("SAMPLE MERGES (first 20)")
print("=" * 50)
for i, merge in enumerate(raw_merges[:20]):
    token1 = bytes(merge[0])
    token2 = bytes(merge[1])
    merged = token1 + token2
    print(f"{i}: {token1!r} + {token2!r} = {merged!r}")

# Load and convert vocab
print("\nLoading vocab...")
with open(DATA_DIR / "ts_vocab.json", "r") as f:
    raw_vocab = json.load(f)
vocab = {int(k): bytes(v) for k, v in raw_vocab.items()}
print(f"Loaded {len(vocab)} vocab entries")

# Print sample vocab
print("\n" + "=" * 50)
print("SAMPLE VOCAB (first 30 entries)")
print("=" * 50)
for i, (token_id, byte_list) in enumerate(list(raw_vocab.items())[:30]):
    token_bytes = bytes(byte_list)
    print(f"ID {token_id}: {token_bytes!r}")

# Create tokenizer
tokenizer = Tokenizer(vocab, merges)

# Test cases
test_texts = [
    "Hello",
    "Hello, world!",
    "The cat sat on the mat.",
    "I am sagar agrawal",
]

print("\n" + "=" * 50)
print("Testing encode:")
print("=" * 50)

for text in test_texts:
    print(f"\nText: {text!r}")
    try:
        encoded = tokenizer.encode(text)
        print(f"Encoded: {encoded}")
        print(f"Num tokens: {len(encoded)}")
        
        # Show what each token represents
        print("Tokens: ", end="")
        for tid in encoded:
            print(f"{vocab[tid]!r} ", end="")
        print()
    except Exception as e:
        print(f"Error: {e}")
