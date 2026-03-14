# BPE Tokenizer: A Complete Guide

## What is a Tokenizer?

A **tokenizer** converts text into numbers (and back) so that machine learning models can process it. Think of it as a translator between human language and computer-readable format.

```
"Hello world" → [15496, 995] → "Hello world"
     ↑              ↑              ↑
   Text         Numbers         Text
  (input)      (for model)    (output)
```

## Why Do We Need Tokenization?

Neural networks can only process numbers, not text. We need to:
1. **Encode**: Convert text → numbers (before feeding to model)
2. **Decode**: Convert numbers → text (after getting model output)

---

## What the Tokenizer Uses

A tokenizer needs two things (pre-trained):

**1. Vocabulary** - A dictionary mapping token IDs to bytes:
```python
vocab = {
    0: b'<|endoftext|>',  # Special token
    1: b'\x00',           # Byte 0
    ...
    256: b' t',           # Merged token: space + t
    257: b'he',           # Merged token: h + e
    258: b' the',         # Merged token: (space+t) + (he)
    ...
}
```

**2. Merges** - An ordered list of byte pairs to combine:
```python
merges = [
    (b' ', b't'),    # Merge #1: space + t → " t"
    (b'h', b'e'),    # Merge #2: h + e → "he"  
    (b' t', b'he'),  # Merge #3: " t" + "he" → " the"
    ...
]
```

---

## Encoding: Text → Numbers

### Step 1: Pre-tokenization

First, split the text into "pre-tokens" using regex patterns. This prevents merging across word boundaries.

```python
text = "The cat sat"

# GPT-2 regex splits on:
# - Contractions ('s, 't, 're, etc.)
# - Words (with optional leading space)
# - Numbers
# - Punctuation
# - Whitespace

pretokens = ["The", " cat", " sat"]
```

### Step 2: Convert to Bytes

Each pre-token becomes a sequence of single-byte objects:

```python
"The"  → b'The'  → [b'T', b'h', b'e']      # 3 bytes
" cat" → b' cat' → [b' ', b'c', b'a', b't'] # 4 bytes
" sat" → b' sat' → [b' ', b's', b'a', b't'] # 4 bytes
```

### Step 3: Apply Merges (in order!)

For each pre-token, apply all merge rules **in the order they were learned**:

**Example: Encoding "the"**

```
Initial:     [b't', b'h', b'e']

Merge #1 (b' ', b't'): No match (no space)
Merge #2 (b'h', b'e'): Match! 
                       [b't', b'h', b'e'] → [b't', b'he']

Merge #3 (b' t', b'he'): No match (no " t")
... continue through all merges ...

Final:       [b't', b'he']
```

**Example: Encoding " the"**

```
Initial:     [b' ', b't', b'h', b'e']

Merge #1 (b' ', b't'): Match!
                       [b' ', b't', b'h', b'e'] → [b' t', b'h', b'e']

Merge #2 (b'h', b'e'): Match!
                       [b' t', b'h', b'e'] → [b' t', b'he']

Merge #3 (b' t', b'he'): Match!
                         [b' t', b'he'] → [b' the']

Final:       [b' the']  ← Single token!
```

### Step 4: Look Up Token IDs

Convert bytes to integer IDs using the vocabulary:

```python
# After merging " the" → [b' the']
# Look up in vocab: b' the' has ID 258

result = [258]
```

**Full Example:**
```python
text = "The cat sat on the mat"

# After encoding:
tokens = [464, 3797, 3332, 319, 262, 2603]
#         The  cat   sat   on  the  mat
```

---

## Decoding: Numbers → Text

Decoding is much simpler—just reverse the lookup!

### Step 1: Look Up Bytes

```python
ids = [464, 3797, 3332, 319, 262, 2603]

# Look up each ID in vocabulary:
bytes_list = [
    vocab[464],   # b'The'
    vocab[3797],  # b' cat'
    vocab[3332],  # b' sat'
    vocab[319],   # b' on'
    vocab[262],   # b' the'
    vocab[2603],  # b' mat'
]
```

### Step 2: Concatenate Bytes

```python
all_bytes = b'The' + b' cat' + b' sat' + b' on' + b' the' + b' mat'
#         = b'The cat sat on the mat'
```

### Step 3: Decode to String

```python
text = all_bytes.decode('utf-8')
# "The cat sat on the mat"
```

**That's it!** Decoding is just: `ID → bytes → concatenate → decode UTF-8`

---

## Special Tokens

Special tokens like `<|endoftext|>` are **never split**. They're handled before pre-tokenization:

```python
text = "Hello<|endoftext|>World"

# Step 1: Split on special tokens
parts = ["Hello", "<|endoftext|>", "World"]

# Step 2: Process each part
# - "Hello" → normal encoding
# - "<|endoftext|>" → look up directly (single token)
# - "World" → normal encoding
```

---

## Visual Summary

### Encoding Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT TEXT                               │
│                    "The cat sat on the mat"                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    1. PRE-TOKENIZATION                           │
│         Split by regex: ["The", " cat", " sat", ...]            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    2. CONVERT TO BYTES                           │
│     "The" → [b'T', b'h', b'e']                                  │
│     " cat" → [b' ', b'c', b'a', b't']                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    3. APPLY MERGES                               │
│     [b'T', b'h', b'e'] → [b'The']                               │
│     [b' ', b'c', b'a', b't'] → [b' cat']                        │
│     (Apply all merge rules in order)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    4. LOOK UP TOKEN IDs                          │
│     b'The' → 464                                                 │
│     b' cat' → 3797                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUT IDs                               │
│                    [464, 3797, 3332, ...]                        │
└─────────────────────────────────────────────────────────────────┘
```

### Decoding Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT IDs                                │
│                    [464, 3797, 3332, ...]                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    1. LOOK UP BYTES                              │
│     464 → b'The'                                                │
│     3797 → b' cat'                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    2. CONCATENATE                                │
│     b'The' + b' cat' + ... = b'The cat sat...'                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    3. DECODE UTF-8                               │
│     b'The cat sat...' → "The cat sat..."                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUT TEXT                              │
│                    "The cat sat on the mat"                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Code Example

```python
from cs336_basics.tokenizer import Tokenizer

# Load a pre-trained tokenizer
tokenizer = Tokenizer.from_files(
    vocab_filepath="data/ts_vocab.json",
    merges_filepath="data/ts_merges.json",
    special_tokens=["<|endoftext|>"]
)

# Encode text to numbers
text = "Once upon a time"
ids = tokenizer.encode(text)
print(f"Encoded: {ids}")
# Encoded: [1234, 5678, 91, 2345]

# Decode numbers back to text
decoded = tokenizer.decode(ids)
print(f"Decoded: {decoded}")
# Decoded: Once upon a time

# Verify roundtrip
assert decoded == text  # ✓ Perfect reconstruction!
```

---

## Key Concepts Summary

| Concept | Description |
|---------|-------------|
| **Vocabulary** | Dictionary mapping token ID → bytes |
| **Merges** | Ordered list of byte pairs to combine |
| **Pre-tokenization** | Split text into chunks (words, punctuation) |
| **Encoding** | Text → bytes → apply merges → look up IDs |
| **Decoding** | IDs → look up bytes → concatenate → decode UTF-8 |
| **Special Tokens** | Never split; handled separately |

---

## FAQ

**Q: Why start with bytes instead of characters?**
A: Bytes handle ANY text (including emojis, Chinese, Arabic) without "unknown" characters. Every possible input can be encoded.

**Q: Why does merge order matter?**
A: The same merges applied in different orders produce different results. We apply them in training order to ensure consistent encoding.

**Q: What if the model outputs invalid byte sequences?**
A: We use `errors='replace'` when decoding, which replaces invalid bytes with the Unicode replacement character (�).

---

## Next Steps

- See `cs336_basics/tokenizer.py` for the implementation
- Run `python scratch/test_encode.py` to see encoding in action
- Run `pytest tests/test_tokenizer.py` to verify correctness
