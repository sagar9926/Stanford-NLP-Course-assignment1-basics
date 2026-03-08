"""BPE Tokenizer implementation."""
from __future__ import annotations

from collections.abc import Iterable, Iterator


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
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.bytes_to_id = { token_bytes : token_id for token_id , token_bytes in vocab.items()}

        # TODO: Build reverse vocab (bytes -> id) for encoding
        # TODO: Build merge priority dict for efficient lookup
        # TODO: Handle special tokens

    def encode(self, text: str) -> list[int]:
        """Convert text to a list of token IDs.

        Args:
            text: The input string to tokenize.

        Returns:
            A list of integer token IDs.
        """
        # TODO: Implement BPE encoding
        # 1. Handle special tokens (extract and preserve them)
        # 2. Apply pretokenization (split on whitespace/punctuation boundaries)
        # 3. For each pretokenized chunk, apply BPE merges
        # 4. Convert merged byte sequences to token IDs
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """Convert a list of token IDs back to a string.

        Args:
            ids: A list of integer token IDs.

        Returns:
            The decoded string.
        """
        # TODO: Implement decoding
        # 1. Look up each ID in vocab to get bytes
        # 2. Concatenate all bytes
        # 3. Decode bytes to string (utf-8)
        raise NotImplementedError

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Memory-efficient streaming tokenization.

        Args:
            iterable: An iterable of strings (e.g., file lines).

        Yields:
            Token IDs one at a time.
        """
        # TODO: Implement streaming encoding
        # Process text incrementally without loading everything into memory
        # Handle tokens that might span across chunk boundaries
        raise NotImplementedError
