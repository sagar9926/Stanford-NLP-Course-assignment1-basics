"""Debug the vocab mismatch in train_bpe_special_tokens test."""
import os
import pickle
from cs336_basics.train_bpe import train_bpe

FIXTURES_PATH = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
SNAPSHOTS_PATH = os.path.join(os.path.dirname(__file__), "..", "tests", "_snapshots")

if __name__ == "__main__":
    input_path = os.path.join(FIXTURES_PATH, "tinystories_sample_5M.txt")
    
    print("Training BPE...")
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )
    
    # Load expected snapshot
    snapshot_path = os.path.join(SNAPSHOTS_PATH, "test_train_bpe_special_tokens.pkl")
    with open(snapshot_path, "rb") as f:
        expected = pickle.load(f)
    
    expected_vocab_values = expected["vocab_values"]
    expected_merges = expected["merges"]
    
    actual_vocab_values = set(vocab.values())
    
    # Find differences in vocab values
    extra_in_actual = actual_vocab_values - expected_vocab_values
    extra_in_expected = expected_vocab_values - actual_vocab_values
    
    print(f"\nVocab values only in OUR output ({len(extra_in_actual)}):")
    for v in sorted(extra_in_actual, key=lambda x: (len(x), x))[:20]:
        print(f"  {v!r}")
    
    print(f"\nVocab values only in EXPECTED ({len(extra_in_expected)}):")
    for v in sorted(extra_in_expected, key=lambda x: (len(x), x))[:20]:
        print(f"  {v!r}")
    
    # Compare merges
    print(f"\nNumber of merges - Ours: {len(merges)}, Expected: {len(expected_merges)}")
    
    # Find first divergence point
    for i, (our_merge, exp_merge) in enumerate(zip(merges, expected_merges)):
        if our_merge != exp_merge:
            print(f"\nFirst merge difference at index {i}:")
            print(f"  Ours:     {our_merge}")
            print(f"  Expected: {exp_merge}")
            
            # Show context around divergence
            print(f"\nMerges around divergence (ours):")
            for j in range(max(0, i-3), min(len(merges), i+4)):
                marker = ">>>" if j == i else "   "
                print(f"  {marker} {j}: {merges[j]}")
            
            print(f"\nMerges around divergence (expected):")
            for j in range(max(0, i-3), min(len(expected_merges), i+4)):
                marker = ">>>" if j == i else "   "
                print(f"  {marker} {j}: {expected_merges[j]}")
            break
    else:
        if len(merges) == len(expected_merges):
            print("\nAll merges match!")
        else:
            print(f"\nMerges match up to index {min(len(merges), len(expected_merges))}")
