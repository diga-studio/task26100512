"""Cross-batch exact fuzzy dedup library."""

from fuzzy_dedup.matcher import Matcher
from fuzzy_dedup.tokenizer import record_elements, tokenize

__all__ = ["Matcher", "record_elements", "tokenize"]
