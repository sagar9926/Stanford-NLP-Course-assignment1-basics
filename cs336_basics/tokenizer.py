"""BPE Tokenizer implementation."""
from __future__ import annotations
import json
import regex as re  # Use regex package for Unicode category support (\p{L}, \p{N})
from collections.abc import Iterable, Iterator

GPT2_PRETOKENIZE_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

class Tokenizer:
    """A BPE (Byte Pair Encoding) tokenizer."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """Initialize the tokenizer with vocabulary, merges, and special tokens.

        Args:
            vocab: Mapping from token ID to token bytes.
            merges: List of BPE merge rules as (token1, token2) tuples.
            special_tokens: List of special tokens that should never be split.
        """
        self.vocab = dict(vocab)  # Make a copy
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        # Add special tokens to vocab if not already present
        for special_token in self.special_tokens:
            special_token_bytes = special_token.encode("utf-8")
            if special_token_bytes not in set(self.vocab.values()):
                self.vocab[len(self.vocab)] = special_token_bytes
        
        # Build reverse mapping: bytes -> token ID
        self.bytes_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        
        # Build regex pattern to split on special tokens (if any)
        if self.special_tokens:
            # Sort by length descending to match longer tokens first
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [re.escape(token) for token in sorted_special]
            self.special_token_pattern = re.compile("(" + "|".join(escaped) + ")")
        else:
            self.special_token_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """Construct a Tokenizer from serialized vocabulary and merges files.

        Args:
            vocab_filepath: Path to the vocabulary JSON file.
            merges_filepath: Path to the merges JSON file.
            special_tokens: Optional list of special tokens.

        Returns:
            A Tokenizer instance.
        """
        # Load vocabulary: {token_id: [byte_values]} -> {int: bytes}
        with open(vocab_filepath, "r") as f:
            raw_vocab = json.load(f)
        vocab = {int(k): bytes(v) for k, v in raw_vocab.items()}
        
        # Load merges: [[bytes1], [bytes2]] -> [(bytes, bytes)]
        with open(merges_filepath, "r") as f:
            raw_merges = json.load(f)
        merges = [(bytes(pair[0]), bytes(pair[1])) for pair in raw_merges]
        
        return cls(vocab, merges, special_tokens)

    def _encode_chunk(self, text: str) -> list[int]:
        """Encode a chunk of text (without special tokens) into token IDs."""
        pretokenized = [
            match.group().encode("utf-8")
            for match in GPT2_PRETOKENIZE_PATTERN.finditer(text)
        ]
        
        result = []
        for pre_token in pretokenized:
            # Convert to list of single-byte bytes objects
            tokens_list = [bytes([b]) for b in pre_token]
            
            # Apply merges in order
            for token1, token2 in self.merges:
                i = 0
                while i < len(tokens_list) - 1:
                    if tokens_list[i] == token1 and tokens_list[i + 1] == token2:
                        tokens_list[i] = token1 + token2
                        del tokens_list[i + 1]
                    else:
                        i += 1
            
            # Convert merged tokens to IDs
            for token in tokens_list:
                result.append(self.bytes_to_id[token])
        
        return result

    def encode(self, text: str) -> list[int]:
        """Convert text to a list of token IDs.

        Args:
            text: The input string to tokenize.

        Returns:
            A list of integer token IDs.
        """
        if not self.special_token_pattern:
            # No special tokens, encode directly
            return self._encode_chunk(text)
        
        # Split text on special tokens, keeping the delimiters
        parts = self.special_token_pattern.split(text)
        
        result = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                # Special token: look up directly
                special_bytes = part.encode("utf-8")
                result.append(self.bytes_to_id[special_bytes])
            else:
                # Regular text: pretokenize and apply merges
                result.extend(self._encode_chunk(part))
        
        return result

    def decode(self, ids: list[int]) -> str:
        """Convert a list of token IDs back to a string.

        Args:
            ids: A list of integer token IDs.

        Returns:
            The decoded string.
        """
        # Concatenate bytes for each token ID
        byte_sequence = b"".join(self.vocab[id] for id in ids)
        # Decode to string, replacing invalid UTF-8 sequences with replacement character
        return byte_sequence.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Memory-efficient streaming tokenization.

        Args:
            iterable: An iterable of strings (e.g., file lines).

        Yields:
            Token IDs one at a time.
        """
        # Buffer to handle tokens that might span chunk boundaries
        buffer = ""
        
        for chunk in iterable:
            buffer += chunk
            
            # If we have special tokens, we need to be careful about boundaries
            if self.special_token_pattern:
                # Find the last special token in the buffer
                parts = self.special_token_pattern.split(buffer)
                
                # Keep the last part in buffer (might be incomplete)
                if len(parts) > 1:
                    # Process all complete parts
                    for part in parts[:-1]:
                        if not part:
                            continue
                        if part in self.special_tokens:
                            special_bytes = part.encode("utf-8")
                            yield self.bytes_to_id[special_bytes]
                        else:
                            for token_id in self._encode_chunk(part):
                                yield token_id
                    buffer = parts[-1]
            else:
                # No special tokens: find a safe split point
                # Split at the last whitespace to avoid breaking tokens
                last_space = buffer.rfind(" ")
                if last_space > 0 and len(buffer) > 1000:  # Only split if buffer is large
                    to_process = buffer[:last_space + 1]
                    buffer = buffer[last_space + 1:]
                    for token_id in self._encode_chunk(to_process):
                        yield token_id
        
        # Process remaining buffer
        if buffer:
            for token_id in self._encode_chunk(buffer):
                yield token_id
