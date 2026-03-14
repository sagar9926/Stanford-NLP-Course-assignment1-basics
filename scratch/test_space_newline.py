"""Test how spaces before newlines are tokenized."""
import regex as re

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

# Test patterns that might create ' ' + '\n' pairs
test_cases = [
    'end of sentence. \n',
    'words \n',
    'hello \n<|endoftext|>',  # What GPT-2 sees before we split
    ' \n',  # Just space+newline
]

print("Testing how space+newline is tokenized:\n")
for test in test_cases:
    tokens = [m.group() for m in PAT.finditer(test)]
    print(f"Input: {repr(test)}")
    print(f"Tokens: {[repr(t) for t in tokens]}")
    # Show byte tuple representation
    for t in tokens:
        as_bytes = t.encode('utf-8')
        as_tuple = tuple(bytes([b]) for b in as_bytes)
        print(f"  {repr(t):15} -> {as_tuple}")
    print()
