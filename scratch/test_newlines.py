"""Test how GPT-2 regex handles newlines."""
import regex as re

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

# Test various newline patterns
test_cases = [
    'hello\n\nworld',
    'hello \n\nworld', 
    '\n\nworld',
    'hello\n\n',
    ' \n',
    '\n ',
    '\n\n',
    'text<|endoftext|>more',  # What happens with special token (not split yet)
]

print("Testing GPT-2 pretokenization pattern with newlines:\n")
for text in test_cases:
    tokens = [m.group() for m in PAT.finditer(text)]
    print(f"{repr(text):30} -> {[repr(t) for t in tokens]}")

# Now test what happens when we split on <|endoftext|> first
print("\n\nSplitting on <|endoftext|> first:")
text_with_special = "Story 1.\n<|endoftext|>\n\nStory 2."
parts = re.split(r"<\|endoftext\|>", text_with_special)
print(f"Original: {repr(text_with_special)}")
print(f"After split: {[repr(p) for p in parts]}")
for i, part in enumerate(parts):
    tokens = [m.group() for m in PAT.finditer(part)]
    print(f"  Part {i} tokens: {[repr(t) for t in tokens]}")
