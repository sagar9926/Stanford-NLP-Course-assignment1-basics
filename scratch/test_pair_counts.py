"""Test the get_pair_counts and merge_pair functions."""
from collections import Counter
from cs336_basics.train_bpe import pretokenize, get_pair_counts, merge_pair


if __name__ == "__main__":
    # Use pretokenize to generate pretokens from text
    text = "hello hello hello hello hello help help"
    pretokens = pretokenize(text)
    print(f"Pretokens from '{text}':")
    print(f"  {pretokens}")
    
    # Build pretoken_counts: convert each pretoken bytes to tuple of single bytes
    pretoken_counts = Counter()
    for pretoken in pretokens:
        # Convert b'hello' -> (b'h', b'e', b'l', b'l', b'o')
        as_tuple = tuple(bytes([b]) for b in pretoken)
        pretoken_counts[as_tuple] += 1
    
    print(f"\nPretoken counts:")
    for pretoken, count in pretoken_counts.items():
        print(f"  {pretoken}: {count}")

    result = get_pair_counts(pretoken_counts)
    print("\nPair counts (top 10):")
    for pair, count in sorted(result.items(), key=lambda x: -x[1])[:10]:
        print(f"  {pair}: {count}")

    # Basic sanity checks
    print("\nRunning sanity checks...")
    assert len(result) > 0, "Should have some pairs"
    assert all(count > 0 for count in result.values()), "All counts should be positive"
    print("All checks passed!")

    # =========================================
    # Test merge_pair
    # =========================================
    print("\n" + "="*50)
    print("Testing merge_pair")
    print("="*50)
    
    # Find the most frequent pair
    best_pair = max(result.items(), key=lambda x: x[1])[0]
    print(f"\nMost frequent pair: {best_pair}")
    
    print(f"\nBefore merge:")
    for pretoken, count in pretoken_counts.items():
        print(f"  {pretoken}: {count}")
    
    # Merge the pair
    merged_counts = merge_pair(pretoken_counts, best_pair)
    
    print(f"\nAfter merging {best_pair} -> {best_pair[0] + best_pair[1]}:")
    for pretoken, count in merged_counts.items():
        print(f"  {pretoken}: {count}")
    
    # Verify the merge
    merged_token = best_pair[0] + best_pair[1]
    print(f"\nVerification:")
    print(f"  Merged token: {merged_token}")
    
    # Check that the pair no longer exists as separate tokens
    for pretoken in merged_counts.keys():
        for i in range(len(pretoken) - 1):
            if pretoken[i] == best_pair[0] and pretoken[i + 1] == best_pair[1]:
                print(f"  ERROR: Pair {best_pair} still exists in {pretoken}")
                break
    else:
        print(f"  OK: Pair {best_pair} has been merged in all pretokens")
    
    # Show new pair counts after merge
    new_pair_counts = get_pair_counts(merged_counts)
    print(f"\nNew pair counts (top 5):")
    for pair, count in sorted(new_pair_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  {pair}: {count}")
