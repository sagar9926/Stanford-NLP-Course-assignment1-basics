"""BPE (Byte Pair Encoding) training implementation."""
from __future__ import annotations

import os
from collections import Counter
from typing import Iterable

import regex as re  # Use regex package for Unicode category support (\p{L}, \p{N})


# GPT-2 pretokenization regex pattern (from tiktoken)
# Matches: contractions, words (with optional leading space), numbers, punctuation, whitespace
GPT2_PRETOKENIZE_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pretokenize(text: str) -> list[bytes]:
    """Split text into pretokens using GPT-2's regex pattern.
    
    Args:
        text: Input text to pretokenize.
        
    Returns:
        List of pretokens as bytes.
    """
    return [match.group().encode("utf-8") for match in GPT2_PRETOKENIZE_PATTERN.finditer(text)]


def get_pair_counts(
    pretoken_counts: dict[tuple[bytes, ...], int]
) -> Counter[tuple[bytes, bytes]]:
    """Count occurrences of adjacent byte pairs across all pretokens.
    
    Args:
        pretoken_counts: Mapping from pretoken (as tuple of bytes) to its count.
        
    Returns:
        Counter of (byte1, byte2) pairs and their total counts.
    """
    pair_counts = Counter()
    for pretoken, count in pretoken_counts.items():
        for i in range(len(pretoken) - 1):
            pair = (pretoken[i], pretoken[i + 1])
            pair_counts[pair] += count
    return pair_counts

def merge_pair(
    pretoken_counts: dict[tuple[bytes, ...], int],
    pair: tuple[bytes, bytes],
) -> dict[tuple[bytes, ...], int]:
    """Merge all occurrences of a pair in the pretoken counts.
    
    Args:
        pretoken_counts: Current pretoken counts.
        pair: The (byte1, byte2) pair to merge.
        
    Returns:
        Updated pretoken counts with the pair merged.
    """
    merged_counts = Counter()
    for pretoken, count in pretoken_counts.items():
        merged_pretoken = []
        i = 0
        while i < len(pretoken):
            if i < len(pretoken) - 1 and (pretoken[i], pretoken[i + 1]) ==pair:
                merged_pretoken.append(pair[0] + pair[1])  # Merge the pair into a single token
                i += 2  # Skip the next byte since it's merged
            else:
                merged_pretoken.append(pretoken[i])
                i += 1
        merged_counts[tuple(merged_pretoken)] += count
    return merged_counts
    

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer on a text corpus.
    
    Args:
        input_path: Path to the training corpus file.
        vocab_size: Target vocabulary size (including special tokens).
        special_tokens: List of special tokens to add to vocabulary.
        
    Returns:
        vocab: Mapping from token ID to token bytes.
        merges: List of merge operations as (token1, token2) tuples.
    """
    # 1. Initialize vocabulary
    vocab = {}
    next_id = 0
    
    # Add special tokens first (get lowest IDs)
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1
    
    # Add all 256 single bytes
    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1
    
    # 2. Read corpus and build pretoken counts
    # Build regex pattern to split on special tokens
    if special_tokens:
        # Escape special regex characters and join with |
        escaped_tokens = [re.escape(token) for token in special_tokens]
        split_pattern = re.compile("|".join(escaped_tokens))
    else:
        split_pattern = None
    
    pretoken_counts = Counter()
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
        
        # Split on special tokens to avoid merging across them
        if split_pattern:
            parts = split_pattern.split(text)
        else:
            parts = [text]
        
        for part in parts:
            for pretoken in pretokenize(part):
                # Convert bytes to tuple of single-byte elements
                as_tuple = tuple(bytes([b]) for b in pretoken)
                pretoken_counts[as_tuple] += 1
    
    # 3. Compute initial pair counts once
    pair_counts = get_pair_counts(pretoken_counts)

    # 4. Iteratively merge until vocab_size is reached
    merges = []
    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        # a. Find most frequent pair (break ties with lexicographically greater pair)
        max_count = max(pair_counts.values())
        best_pair = max(
            (pair for pair, count in pair_counts.items() if count == max_count),
        )

        # b. Merge pair in all pretokens, incrementally updating pair_counts
        merged_token = best_pair[0] + best_pair[1]
        new_pretoken_counts: dict[tuple[bytes, ...], int] = {}

        for old_pretoken, count in pretoken_counts.items():
            # Apply merge
            new_pretoken_list = []
            i = 0
            changed = False
            while i < len(old_pretoken):
                if (i < len(old_pretoken) - 1
                        and old_pretoken[i] == best_pair[0]
                        and old_pretoken[i + 1] == best_pair[1]):
                    new_pretoken_list.append(merged_token)
                    i += 2
                    changed = True
                else:
                    new_pretoken_list.append(old_pretoken[i])
                    i += 1
            new_pretoken = tuple(new_pretoken_list)

            if changed:
                # Subtract old pair counts
                for i in range(len(old_pretoken) - 1):
                    old_pair = (old_pretoken[i], old_pretoken[i + 1])
                    pair_counts[old_pair] -= count
                    if pair_counts[old_pair] <= 0:
                        del pair_counts[old_pair]
                # Add new pair counts
                for i in range(len(new_pretoken) - 1):
                    new_pair = (new_pretoken[i], new_pretoken[i + 1])
                    pair_counts[new_pair] = pair_counts.get(new_pair, 0) + count

            new_pretoken_counts[new_pretoken] = new_pretoken_counts.get(new_pretoken, 0) + count

        pretoken_counts = new_pretoken_counts

        # c. Add merged token to vocabulary
        vocab[next_id] = merged_token
        next_id += 1

        # d. Record the merge
        merges.append(best_pair)

    return vocab, merges
