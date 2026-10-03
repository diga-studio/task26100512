"""Exact fuzzy dedup across import batches."""

from .matcher import DedupMatcher
from .tokenizer import record_elements, tokenize_value

__all__ = ["DedupMatcher", "record_elements", "tokenize_value"]
