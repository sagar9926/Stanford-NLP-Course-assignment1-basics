"""Test the train_bpe function."""
from cs336_basics.train_bpe import train_bpe
import os

# Use a small test corpus
FIXTURES_PATH = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
corpus_path = os.path.join(FIXTURES_PATH, "corpus.en")

if __name__ == "__main__":
    print(f"Training BPE on: {corpus_path}")
    print(f"Vocab size: 300")
    print(f"Special tokens: ['<|endoftext|>']")
    print()
    
    vocab, merges = train_bpe(
        input_path=corpus_path,
        vocab_size=300,
        special_tokens=["<|endoftext|>"],
    )
    
    print(f"Vocab size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    
    print(f"\nFirst 10 vocab entries:")
    for i, (idx, token) in enumerate(sorted(vocab.items())[:10]):
        print(f"  {idx}: {token}")
    
    print(f"\nFirst 10 merges:")
    for i, merge in enumerate(merges[:10]):
        print(f"  {i}: {merge[0]} + {merge[1]} -> {merge[0] + merge[1]}")
    
    print(f"\nLast 5 merges (most complex tokens):")
    for i, merge in enumerate(merges[-5:], len(merges) - 5):
        print(f"  {i}: {merge[0]} + {merge[1]} -> {merge[0] + merge[1]}")
