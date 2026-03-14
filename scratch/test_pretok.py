"""Test the pretokenization pattern."""
import regex as re

# GPT-2 pretokenization regex pattern (from tiktoken)
GPT2_PRETOKENIZE_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pretokenize(text: str) -> list[bytes]:
    """Split text into pretokens using GPT-2's regex pattern."""
    # Note: Your code had `raise re.finditer(...)` which is wrong
    # Should be `return [match.encode(...) for match in ...]`
    return [match.group().encode("utf-8") for match in GPT2_PRETOKENIZE_PATTERN.finditer(text)]


# Test cases
if __name__ == "__main__":
    test_cases = [
        "some text that i'll pre-tokenize",
        "Hello, how are you?",
        "i'll go",
        "Héllò hôw are ü? 🙃",
        "   multiple   spaces   ",
        "numbers123and456words",
        "<|endoftext|>",
    ]
    
    for text in test_cases:
        print(f"Input: {repr(text)}")
        tokens = pretokenize(text)
        print(f"  Tokens (bytes): {tokens}")
        print(f"  Tokens (str):   {[t.decode('utf-8') for t in tokens]}")
        print()
