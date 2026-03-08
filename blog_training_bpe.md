# Building a BPE Tokenizer from Scratch: A Complete Guide

*A deep dive into byte-level Byte Pair Encoding — the tokenization algorithm behind GPT-2, GPT-3, and GPT-4.*

---

## Table of Contents
1. [Introduction](#introduction)
2. [Understanding Unicode and UTF-8](#understanding-unicode-and-utf-8)
3. [Why BPE? The Tokenization Problem](#why-bpe-the-tokenization-problem)
4. [Step-by-Step BPE Training](#step-by-step-bpe-training)
5. [Implementation Deep Dive](#implementation-deep-dive)
6. [Optimization Techniques](#optimization-techniques)
7. [Parallelizing Pre-tokenization](#parallelizing-pre-tokenization)
8. [Real-World Results](#real-world-results)
9. [Common Pitfalls](#common-pitfalls)
10. [Key Takeaways](#key-takeaways)

---

## Introduction

Before a language model can process a single word, it needs a **tokenizer** — an algorithm that converts raw text into a sequence of integer IDs. The tokenizer is the very first and very last component in the pipeline: text goes in, numbers come out, and the model never sees characters directly.

```
"Hello, world!" → Tokenizer → [15496, 11, 995, 0] → Model → [...]
```

GPT-2, GPT-3, GPT-4, and many other modern LLMs use a tokenization algorithm called **Byte Pair Encoding (BPE)**. Despite its importance, BPE is often glossed over in tutorials. In this post, we'll build a byte-level BPE tokenizer from scratch in Python, explain every design decision, and train it on a real dataset.

**By the end, you'll understand:**
- How Unicode and UTF-8 encodings work (and why they matter)
- Why BPE exists and what problem it solves
- How pretokenization works (and why GPT-2 uses a specific regex)
- The core BPE training loop, step by step with a worked example
- How to optimize from a naive implementation to production-ready code
- How to parallelize for large datasets
- Common mistakes and how to avoid them

---

## Understanding Unicode and UTF-8

Before diving into BPE, we need to understand how text is represented in computers.

### Unicode: A Universal Character Set

**Unicode** is a standard that assigns a unique number (called a **code point**) to every character in every language. As of Unicode 16.0, there are 154,998 characters across 168 scripts.

```python
>>> ord('A')      # ASCII letter
65
>>> ord('牛')     # Chinese character
29275
>>> chr(29275)    # Convert back
'牛'
>>> ord('😀')     # Emoji
128512
```

### The Problem with Raw Unicode

If we tried to tokenize raw Unicode code points directly:
- Vocabulary would be ~155K items (too large!)
- Most characters are extremely rare (sparse vocabulary)
- Different languages would have vastly different token counts for the same meaning

### UTF-8: A Byte-Level Encoding

**UTF-8** converts Unicode code points into sequences of 1-4 bytes:

| Character | Code Point | UTF-8 Bytes | Byte Values |
|-----------|------------|-------------|-------------|
| `'A'` | 65 | 1 byte | `[65]` |
| `'é'` | 233 | 2 bytes | `[195, 169]` |
| `'牛'` | 29275 | 3 bytes | `[231, 137, 155]` |
| `'😀'` | 128512 | 4 bytes | `[240, 159, 152, 128]` |

```python
>>> "hello! こんにちは!".encode("utf-8")
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'

>>> list("hello! こんにちは!".encode("utf-8"))
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, ...]
```

**Why UTF-8 for BPE?**
- Only 256 possible byte values → manageable initial vocabulary
- Variable-length encoding → efficient for ASCII-heavy text (English)
- No "unknown" tokens — any input can be represented as bytes
- UTF-8 is the dominant web encoding (98%+ of websites)

### A Common Mistake: Decoding Single Bytes

This function is **WRONG**:

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```

**Why?** Multi-byte characters can't be decoded one byte at a time:

```python
>>> decode_utf8_bytes_to_str_wrong("牛".encode("utf-8"))
# UnicodeDecodeError! Each byte alone is invalid UTF-8
```

The correct approach decodes all bytes together: `bytestring.decode("utf-8")`

---

## Why BPE? The Tokenization Problem

Language models need a fixed vocabulary — a finite set of "tokens" they understand. But language is messy:

| Approach | Vocabulary Size | Sequence Length | Problem |
|----------|----------------|-----------------|---------|
| **Character-level** | ~256 bytes | Very long | Model must "learn" spelling; long-range dependencies |
| **Word-level** | 100K+ words | Short | Huge vocabulary; can't handle typos, new words, rare words |
| **Subword (BPE)** | ~10K-50K | Medium | Best of both worlds ✓ |

**BPE offers a middle ground.** It starts with individual bytes (256 tokens) and **iteratively merges the most frequent adjacent pair** into a new token. Common words like `" the"` become a single token. Rare words are broken into subword pieces. The model never encounters an "unknown" token — any byte sequence can be represented.

---

## Step-by-Step BPE Training: A Worked Example

Before diving into code, let's walk through BPE training manually on a small corpus. This will build intuition for what the algorithm does.

### Example Corpus

```
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```

**Vocabulary:** We start with `<|endoftext|>` (special token) + 256 byte characters.

### Step 1: Pre-tokenize and Count

For simplicity, assume we split on whitespace (real BPE uses a regex):

| Pre-token | Count |
|-----------|-------|
| `(l, o, w)` | 5 |
| `(l, o, w, e, r)` | 2 |
| `(w, i, d, e, s, t)` | 3 |
| `(n, e, w, e, s, t)` | 6 |

### Step 2: Count All Adjacent Pairs

| Pair | Count | Appears in... |
|------|-------|---------------|
| `(l, o)` | 5 + 2 = **7** | low, lower |
| `(o, w)` | 5 + 2 = **7** | low, lower |
| `(w, e)` | 2 + 6 = **8** | lower, newest |
| `(e, r)` | **2** | lower |
| `(w, i)` | **3** | widest |
| `(i, d)` | **3** | widest |
| `(d, e)` | **3** | widest |
| `(e, s)` | 3 + 6 = **9** | widest, newest |
| `(s, t)` | 3 + 6 = **9** | widest, newest |
| `(n, e)` | **6** | newest |
| `(e, w)` | **6** | newest |

### Step 3: Merge the Most Frequent Pair

Pairs `(e, s)` and `(s, t)` are tied at count 9. We pick the **lexicographically greater** one: `(s, t)`.

**After merge 1:** `st` becomes a single token.

| Pre-token | Count |
|-----------|-------|
| `(l, o, w)` | 5 |
| `(l, o, w, e, r)` | 2 |
| `(w, i, d, e, st)` | 3 |
| `(n, e, w, e, st)` | 6 |

### Step 4: Repeat

**Merge 2:** `(e, st)` with count 9 → `est`

| Pre-token | After merge |
|-----------|-------------|
| `(w, i, d, est)` | 3 |
| `(n, e, w, est)` | 6 |

**Merge 3:** `(o, w)` with count 7 → `ow`

**Merge 4:** `(l, ow)` with count 7 → `low`

**Merge 5:** `(w, est)` with count 9 → `west`

**Merge 6:** `(n, e)` with count 6 → `ne`

### Final State (after 6 merges)

**Vocabulary:** `[<|endoftext|>, 256 bytes, st, est, ow, low, west, ne]`

**Merges list:** `[(s,t), (e,st), (o,w), (l,ow), (w,est), (n,e)]`

Now `"newest"` tokenizes as `[ne, west]` — just 2 tokens instead of 6!

---

## Implementation Deep Dive

Now let's implement this in Python.

### The Imports

```python
from __future__ import annotations

import os
from collections import Counter

import regex as re
```

We use the `regex` package (not Python's built-in `re`) because it supports Unicode categories like `\p{L}` (any letter) and `\p{N}` (any number), which the GPT-2 pretokenization pattern requires.

---

### Step 1: Pretokenization — Splitting Text Before Training

Before we run BPE, we **pretokenize** the text: split it into coarse chunks using a regex pattern. This prevents BPE from merging across word boundaries. For example, we don't want `"end of"` to merge into a single token that spans two words.

GPT-2 uses this specific regex pattern:

```python
GPT2_PRETOKENIZE_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
```

Let's break this down piece by piece:

| Pattern | What it matches | Example |
|---|---|---|
| `'(?:[sdmt]\|ll\|ve\|re)` | English contractions | `'s`, `'t`, `'ll`, `'ve`, `'re` |
| `?\p{L}+` | Words (with optional leading space) | `" Hello"`, `"world"` |
| `?\p{N}+` | Numbers (with optional leading space) | `" 42"`, `"2024"` |
| `?[^\s\p{L}\p{N}]+` | Punctuation/symbols | `" !"`, `"..."` |
| `\s+(?!\S)` | Trailing whitespace (not followed by non-whitespace) | `"   "` at end of line |
| `\s+` | Other whitespace | `" "`, `"\n"` |

The pretokenize function applies this pattern and returns chunks as **bytes** (not strings):

```python
def pretokenize(text: str) -> list[bytes]:
    return [
        match.group().encode("utf-8")
        for match in GPT2_PRETOKENIZE_PATTERN.finditer(text)
    ]
```

**Why bytes?** Because this is a *byte-level* BPE tokenizer. Working in bytes means every possible input can be tokenized — even binary data or invalid UTF-8. The initial vocabulary is simply all 256 possible byte values.

### Quick example:

```python
>>> pretokenize("Hello, world! It's nice.")
[b'Hello', b',', b' world', b'!', b" It", b"'s", b' nice', b'.']
```

Notice how `" world"` keeps its leading space — BPE will learn that "space + word" is a common pattern and merge them together.

---

## Step 2: Counting Byte Pairs

Once we have pretokens, we need to count every adjacent pair of bytes across the entire corpus. Each pretoken is represented as a **tuple of byte chunks**, and we weight by how often each pretoken appears.

```python
def get_pair_counts(
    pretoken_counts: dict[tuple[bytes, ...], int]
) -> Counter[tuple[bytes, bytes]]:
    pair_counts = Counter()
    for pretoken, count in pretoken_counts.items():
        for i in range(len(pretoken) - 1):
            pair = (pretoken[i], pretoken[i + 1])
            pair_counts[pair] += count
    return pair_counts
```

**Key insight:** We don't iterate over the raw corpus. Instead, we keep a dictionary mapping each unique pretoken to its frequency. If `(b'H', b'e', b'l', b'l', b'o')` appears 1,000 times, we count its pairs 1,000 times without scanning 1,000 copies.

### Example:

Given `pretoken_counts = {(b'a', b'b', b'c'): 5, (b'a', b'b'): 3}`:

- Pair `(b'a', b'b')` → count = 5 + 3 = **8**
- Pair `(b'b', b'c')` → count = **5**

---

## Step 3: Merging a Pair

When we find the most frequent pair, we merge it everywhere:

```python
def merge_pair(
    pretoken_counts: dict[tuple[bytes, ...], int],
    pair: tuple[bytes, bytes],
) -> dict[tuple[bytes, ...], int]:
    merged_counts = Counter()
    for pretoken, count in pretoken_counts.items():
        merged_pretoken = []
        i = 0
        while i < len(pretoken):
            if i < len(pretoken) - 1 and (pretoken[i], pretoken[i + 1]) == pair:
                merged_pretoken.append(pair[0] + pair[1])  # Concatenate bytes
                i += 2  # Skip both elements
            else:
                merged_pretoken.append(pretoken[i])
                i += 1
        merged_counts[tuple(merged_pretoken)] += count
    return merged_counts
```

For example, merging `(b'e', b' ')` in `(b'h', b'e', b' ', b'w')` produces `(b'h', b'e ', b'w')`. The two bytes become one token.

---

## Step 4: The Main Training Loop

Now we put it all together. The `train_bpe` function orchestrates everything:

### 4a: Initialize the Vocabulary

```python
vocab = {}
next_id = 0

# Special tokens get the lowest IDs
for token in special_tokens:
    vocab[next_id] = token.encode("utf-8")
    next_id += 1

# All 256 single bytes
for i in range(256):
    vocab[next_id] = bytes([i])
    next_id += 1
```

The vocabulary starts with **special tokens** (like `<|endoftext|>`) followed by all 256 byte values. This guarantees any input can be tokenized, since every byte has an ID.

### 4b: Read the Corpus and Build Pretoken Counts

```python
# Split text on special tokens first
if special_tokens:
    escaped_tokens = [re.escape(token) for token in special_tokens]
    split_pattern = re.compile("|".join(escaped_tokens))

pretoken_counts = Counter()
with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()
    
    if split_pattern:
        parts = split_pattern.split(text)
    else:
        parts = [text]
    
    for part in parts:
        for pretoken in pretokenize(part):
            as_tuple = tuple(bytes([b]) for b in pretoken)
            pretoken_counts[as_tuple] += 1
```

**Why split on special tokens?** We don't want BPE to merge bytes *across* a special token boundary. If the corpus contains `"...end<|endoftext|>Once..."`, the `"d"` from `"end"` should never merge with `"<"` from the special token.

Each pretoken is stored as a tuple of single bytes: `b"the"` → `(b't', b'h', b'e')`. The Counter tracks how often each unique pretoken appears—this avoids redundant work later.

### 4c: The Iterative Merge Loop (Naive Version)

The simplest version recomputes pair counts from scratch each iteration:

```python
merges = []
while len(vocab) < vocab_size:
    pair_counts = get_pair_counts(pretoken_counts)   # O(total tokens)
    if not pair_counts:
        break
    
    max_count = max(pair_counts.values())
    best_pair = max(
        (pair for pair, count in pair_counts.items() if count == max_count),
    )
    
    pretoken_counts = merge_pair(pretoken_counts, best_pair)
    
    merged_token = best_pair[0] + best_pair[1]
    vocab[next_id] = merged_token
    next_id += 1
    merges.append(best_pair)
```

**Tie-breaking:** When multiple pairs have the same frequency, we pick the lexicographically greatest pair. This matches the reference implementation's behavior.

**The problem:** This is $O(N \times M)$ where $N$ is the number of merge steps and $M$ is the total number of tokens across all pretokens. For a 10,000-token vocabulary on TinyStories, this takes ~3 seconds on a small corpus and would be impractical on larger datasets.

---

## Step 5: The Optimization — Incremental Updates

The key insight: **after each merge, only the pairs overlapping with the merged pair change.** We don't need to recount everything.

```python
# Compute pair counts ONCE
pair_counts = get_pair_counts(pretoken_counts)

merges = []
while len(vocab) < vocab_size:
    if not pair_counts:
        break

    # Find most frequent pair
    max_count = max(pair_counts.values())
    best_pair = max(
        (pair for pair, count in pair_counts.items() if count == max_count),
    )

    # Merge and incrementally update counts
    merged_token = best_pair[0] + best_pair[1]
    new_pretoken_counts = {}

    for old_pretoken, count in pretoken_counts.items():
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

        new_pretoken_counts[new_pretoken] = (
            new_pretoken_counts.get(new_pretoken, 0) + count
        )

    pretoken_counts = new_pretoken_counts

    vocab[next_id] = merged_token
    next_id += 1
    merges.append(best_pair)
```

**What changed:**
1. `get_pair_counts` runs **once** at the start, not every iteration.
2. For each pretoken, we check if the merge actually applies (`changed` flag).
3. If it does, we **subtract** the old pairs and **add** the new ones — an incremental update.
4. If it doesn't, we skip the update entirely.

This brings the complexity from $O(N \times M)$ down to roughly $O(N \times U)$ where $U$ is the number of *unique* pretokens — a massive improvement.

**Benchmark on a test corpus (vocab_size=500):**

| Implementation | Time |
|---|---|
| Naive (recount every iteration) | ~3.0s |
| Optimized (incremental updates) | ~0.4s |

---

## Parallelizing Pre-tokenization

For large datasets like TinyStories (2.1GB), pre-tokenization becomes the bottleneck. The solution: **parallelize across multiple CPU cores**.

### The Strategy

1. **Chunk the corpus** at special token boundaries (e.g., `<|endoftext|>`)
2. **Process chunks in parallel** using `multiprocessing`
3. **Combine** the pretoken counts from all workers

### Why Split at Special Token Boundaries?

We never want to merge bytes across document boundaries. Since `<|endoftext|>` separates documents, splitting there is always valid.

### Implementation

```python
from multiprocessing import Pool
import regex as re

def find_chunk_boundaries(text: str, special_token: str, num_chunks: int) -> list[int]:
    """Find positions to split text at special token boundaries."""
    chunk_size = len(text) // num_chunks
    boundaries = [0]
    
    for i in range(1, num_chunks):
        target = i * chunk_size
        # Find the nearest special token after the target position
        pos = text.find(special_token, target)
        if pos != -1:
            boundaries.append(pos + len(special_token))
        else:
            boundaries.append(len(text))
    
    boundaries.append(len(text))
    return boundaries

def pretokenize_chunk(chunk: str, split_pattern: re.Pattern) -> Counter:
    """Pre-tokenize a single chunk and return pretoken counts."""
    pretoken_counts = Counter()
    
    if split_pattern:
        parts = split_pattern.split(chunk)
    else:
        parts = [chunk]
    
    for part in parts:
        for match in GPT2_PRETOKENIZE_PATTERN.finditer(part):
            pretoken = match.group().encode("utf-8")
            as_tuple = tuple(bytes([b]) for b in pretoken)
            pretoken_counts[as_tuple] += 1
    
    return pretoken_counts

def parallel_pretokenize(text: str, special_tokens: list[str], num_workers: int = 8) -> Counter:
    """Pre-tokenize text in parallel across multiple workers."""
    # Build split pattern
    if special_tokens:
        escaped = [re.escape(t) for t in special_tokens]
        split_pattern = re.compile("|".join(escaped))
    else:
        split_pattern = None
    
    # Find chunk boundaries
    boundaries = find_chunk_boundaries(text, special_tokens[0], num_workers)
    chunks = [text[boundaries[i]:boundaries[i+1]] for i in range(len(boundaries)-1)]
    
    # Process in parallel
    with Pool(num_workers) as pool:
        results = pool.starmap(
            pretokenize_chunk, 
            [(chunk, split_pattern) for chunk in chunks]
        )
    
    # Combine results
    total_counts = Counter()
    for result in results:
        total_counts += result
    
    return total_counts
```

### Performance Impact

| Approach | Time (TinyStories) |
|----------|-------------------|
| Single-threaded | ~8-10 minutes |
| Parallelized (8 cores) | ~1-2 minutes |

**Note:** The merge loop itself cannot be parallelized in Python due to the Global Interpreter Lock (GIL) — each merge depends on the previous one's result.

---

## Real-World Results: Training on TinyStories

We trained our BPE tokenizer on the TinyStories dataset (~2.1M short children's stories) with `vocab_size=10,000`:

```python
vocab, merges = train_bpe(
    input_path="data/TinyStoriesV2-GPT4-train.txt",
    vocab_size=10_000,
    special_tokens=["<|endoftext|>"],
)
```

### Results

| Metric | Value |
|---|---|
| Training time | ~1.3 hours |
| Peak memory | ~14.5 GB |
| Final vocab size | 10,000 |
| Number of merges | 9,743 |
| Longest token | `" accomplishment"` (15 bytes) |

The longest token, `" accomplishment"` (note the leading space), makes intuitive sense. TinyStories contains simple children's narratives where characters often learn about accomplishments. BPE discovered that this entire word (including its preceding space) appears frequently enough to justify a dedicated vocabulary slot.

### What the first few merges look like:

The very first merges are the most common byte pairs in English text:

1. `(b'e', b' ')` → `b'e '` — the letter "e" followed by a space
2. `(b't', b'h')` → `b'th'` — the beginning of "the", "that", "they", etc.
3. `(b'i', b'n')` → `b'in'` — appears in "in", "ing", "int", etc.

These gradually build up to full words and eventually common phrases with leading spaces.

---

## The Complete Picture

Here's how all the pieces fit together:

```
                        Raw text
                            │
                            ▼
            ┌───────────────────────────────┐
            │  Split on special tokens       │  "Hello<|endoftext|>World"
            │  (prevent cross-doc merging)   │  → ["Hello", "World"]
            └───────────────────────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │  Pretokenize (GPT-2 regex)     │  "Hello World"
            │  - Contractions: 's, 'll       │  → [b"Hello", b" World"]
            │  - Words with leading space    │
            │  - Numbers, punctuation        │
            └───────────────────────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │  Convert to byte tuples        │  b"Hello"
            │  + count occurrences           │  → (b'H', b'e', b'l', b'l', b'o')
            └───────────────────────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │  BPE merge loop                │
            │  1. Count all byte pairs       │
            │  2. Find most frequent pair    │
            │  3. Merge it everywhere        │
            │  4. Update counts incrementally│
            │  5. Repeat until vocab_size    │
            └───────────────────────────────┘
                            │
                            ▼
                    vocab + merges
```

---

## Common Pitfalls

Here are mistakes students frequently make when implementing BPE:

### 1. Merging Across Special Token Boundaries

**Wrong:** Pre-tokenizing the entire text without splitting on special tokens.

```python
# WRONG: This allows merging across document boundaries
for pretoken in pretokenize(entire_text):
    ...
```

**Right:** Split on special tokens first, then pre-tokenize each part.

```python
# RIGHT: Documents are isolated
parts = re.split("|".join([re.escape(t) for t in special_tokens]), text)
for part in parts:
    for pretoken in pretokenize(part):
        ...
```

### 2. Wrong Tie-Breaking Order

When multiple pairs have the same count, you must pick the **lexicographically greater** pair.

```python
# WRONG: Arbitrary order
best_pair = max(pair_counts, key=pair_counts.get)

# RIGHT: Break ties lexicographically
max_count = max(pair_counts.values())
best_pair = max(p for p, c in pair_counts.items() if c == max_count)
```

### 3. Forgetting That Bytes Are Not Characters

In Python, there's no single-byte type. A "single byte" is still a `bytes` object:

```python
>>> type(b"hello"[0])
<class 'int'>  # Indexing returns an int!

>>> type(bytes([65]))
<class 'bytes'>  # Use bytes([...]) for a single-byte bytes object
```

When storing pretokens as tuples, each element should be `bytes`, not `int`:

```python
# WRONG
as_tuple = tuple(pretoken)  # tuple of ints!

# RIGHT
as_tuple = tuple(bytes([b]) for b in pretoken)  # tuple of bytes
```

### 4. Not Escaping Special Tokens in Regex

Special tokens often contain regex metacharacters (like `|`). Always escape them:

```python
# WRONG: "|" in special token breaks the pattern
pattern = "|".join(special_tokens)

# RIGHT: Escape first
pattern = "|".join(re.escape(t) for t in special_tokens)
```

### 5. Modifying Dictionary While Iterating

When updating pair counts, don't delete from the dict you're iterating over:

```python
# WRONG: RuntimeError!
for pair in pair_counts:
    if pair_counts[pair] <= 0:
        del pair_counts[pair]

# RIGHT: Collect keys first, or use a check
if pair_counts[pair] <= 0:
    del pair_counts[pair]  # OK if not in a loop over pair_counts
```

### 6. Inefficient String Building

Each merge produces a new merged token. Don't create strings — stay in bytes:

```python
# SLOW: String operations
merged = str(pair[0]) + str(pair[1])

# FAST: Direct byte concatenation
merged = pair[0] + pair[1]  # bytes + bytes = bytes
```

---

## Key Takeaways

1. **BPE is a compression algorithm applied to tokenization.** It finds the most common byte patterns and assigns them single tokens, achieving a balance between vocabulary size and sequence length.

2. **Byte-level means universal.** Starting from raw bytes (not characters) means the tokenizer can handle any input — any language, any encoding, even binary data. UTF-8 is the encoding of choice because it's variable-length (efficient for ASCII) and dominant on the web.

3. **Pretokenization matters.** Without it, BPE would merge across word boundaries, creating nonsensical tokens like `"d O"` (end of one word + start of next). The GPT-2 regex carefully handles contractions, numbers, punctuation, and whitespace.

4. **Special tokens need special handling.** They're added to the vocabulary first (lowest IDs) and excluded from the merging process. Always split on special tokens before pre-tokenization to prevent cross-document merging.

5. **Incremental updates are essential.** The naive approach recomputes all pair counts every iteration ($O(N \times M)$). By caching counts and updating only what changes, training becomes ~7x faster ($O(N \times U)$).

6. **Parallelization for pre-tokenization.** The bottleneck on large datasets is often pre-tokenization. Parallelize by chunking at special token boundaries. The merge loop cannot be parallelized due to sequential dependencies.

7. **Tie-breaking is deterministic.** When pairs have equal counts, always pick the lexicographically greater pair for reproducible results across implementations.

---

## Profiling Your Implementation

Before optimizing, **profile first** to find the actual bottleneck:

### Using cProfile

```python
import cProfile
import pstats

cProfile.run('train_bpe("data/corpus.txt", 1000, ["<|endoftext|>"])', 'bpe.prof')

# Analyze
stats = pstats.Stats('bpe.prof')
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### Using scalene (Recommended)

```bash
pip install scalene
scalene train_bpe.py
```

Scalene shows CPU time, memory usage, and even GPU usage in a nice HTML report.

### What to Look For

| Bottleneck | Symptom | Solution |
|------------|---------|----------|
| Pre-tokenization | 70%+ time in `finditer` | Parallelize with multiprocessing |
| Pair counting | High time in `get_pair_counts` | Use incremental updates |
| Finding max | Time in `max()` calls | Consider a priority queue (heap) |
| Memory | Peak > 30GB | Process corpus in streaming chunks |

---

## Further Reading

- [Sennrich et al. (2016)](https://arxiv.org/abs/1508.07909) — The original BPE for NLP paper
- [Radford et al. (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2 paper introducing byte-level BPE
- [Kudo & Richardson (2018)](https://arxiv.org/abs/1804.10959) — SentencePiece and Unigram tokenization
- [HuggingFace Tokenizers](https://huggingface.co/docs/tokenizers/) — Production-grade tokenizer library in Rust
- [tiktoken](https://github.com/openai/tiktoken) — OpenAI's fast BPE implementation

---

## Appendix: Quick Reference

### Vocabulary Structure

```python
vocab = {
    0: b'<|endoftext|>',     # Special tokens first
    1: b'\x00',               # Byte 0
    2: b'\x01',               # Byte 1
    ...
    256: b'\xff',             # Byte 255
    257: b'th',               # First merge
    258: b'the',              # Second merge
    ...
}
```

### Merges List Structure

```python
merges = [
    (b't', b'h'),            # First merge: t + h → th
    (b'th', b'e'),           # Second merge: th + e → the
    (b'e', b' '),            # Third merge: e + space → "e "
    ...
]
```

### Complete Function Signature

```python
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.
    
    Args:
        input_path: Path to training text file
        vocab_size: Target vocabulary size (including special tokens + 256 bytes)
        special_tokens: Tokens to preserve atomically (e.g., ["<|endoftext|>"])
    
    Returns:
        vocab: Mapping from token ID to token bytes
        merges: List of (token1, token2) merge operations in order
    """
```

---

*This post was written as part of Stanford CS336: Language Modeling from Scratch. The complete implementation is available on GitHub.*
