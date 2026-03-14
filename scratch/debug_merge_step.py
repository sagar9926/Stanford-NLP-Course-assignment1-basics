"""Debug specific merge step where divergence happens."""
import os
import pickle
from collections import Counter
from cs336_basics.train_bpe import pretokenize, get_pair_counts, merge_pair
import regex as re

FIXTURES_PATH = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
SNAPSHOTS_PATH = os.path.join(os.path.dirname(__file__), "..", "tests", "_snapshots")

if __name__ == "__main__":
    input_path = os.path.join(FIXTURES_PATH, "tinystories_sample_5M.txt")
    special_tokens = ["<|endoftext|>"]
    
    # Load expected merges
    snapshot_path = os.path.join(SNAPSHOTS_PATH, "test_train_bpe_special_tokens.pkl")
    with open(snapshot_path, "rb") as f:
        expected = pickle.load(f)
    expected_merges = expected["merges"]
    
    # Build initial pretoken counts (same as train_bpe)
    if special_tokens:
        escaped_tokens = [re.escape(token) for token in special_tokens]
        split_pattern = re.compile("|".join(escaped_tokens))
    else:
        split_pattern = None
    
    pretoken_counts = Counter()
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if split_pattern:
                parts = split_pattern.split(line)
            else:
                parts = [line]
            
            for part in parts:
                for pretoken in pretokenize(part):
                    as_tuple = tuple(bytes([b]) for b in pretoken)
                    pretoken_counts[as_tuple] += 1
    
    print(f"Initial pretoken count: {len(pretoken_counts)}")
    
    # Run merges until step 297 (where divergence happens)
    merges_done = []
    for i in range(297):
        pair_counts = get_pair_counts(pretoken_counts)
        if not pair_counts:
            break
        
        max_count = max(pair_counts.values())
        best_pair = max(
            (pair for pair, count in pair_counts.items() if count == max_count),
        )
        pretoken_counts = merge_pair(pretoken_counts, best_pair)
        merges_done.append(best_pair)
        
        # Check if we still match expected
        if best_pair != expected_merges[i]:
            print(f"DIVERGENCE at step {i}!")
            print(f"  Ours:     {best_pair}")
            print(f"  Expected: {expected_merges[i]}")
            break
    
    # Now at step 297, show the pair counts
    print(f"\n=== At merge step 297 ===")
    pair_counts = get_pair_counts(pretoken_counts)
    
    # Find the pairs mentioned
    pair_space_newline = (b' ', b'\n')
    pair_space_n_o = (b' n', b'o')
    pair_newline_newline = (b'\n', b'\n')
    
    print(f"Pair counts:")
    print(f"  (b' ', b'\\n'):  {pair_counts.get(pair_space_newline, 0)}")
    print(f"  (b' n', b'o'):  {pair_counts.get(pair_space_n_o, 0)}")  
    print(f"  (b'\\n', b'\\n'): {pair_counts.get(pair_newline_newline, 0)}")
    
    # Show top pairs
    print(f"\nTop 20 pairs by count:")
    for pair, count in sorted(pair_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {count}: {pair}")
    
    # What did expected merge at this step?
    print(f"\nExpected merge at 297: {expected_merges[297]}")
