"""Tokenization: NFKC + casefold, then maximal runs of str.isalnum() chars."""

from __future__ import annotations

import unicodedata


def tokenize(value: str | None) -> frozenset[str]:
    """Split a field value into a set of unique words.

    The value is normalized with NFKC, then casefolded, then split into
    maximal runs of characters for which str.isalnum() is true. null and
    empty values produce no words. Duplicate words count once (set semantics).
    """
    if not value:
        return frozenset()
    text = unicodedata.normalize("NFKC", value).casefold()
    words: set[str] = set()
    start = -1
    for index, char in enumerate(text):
        if char.isalnum():
            if start < 0:
                start = index
        elif start >= 0:
            words.add(text[start:index])
            start = -1
    if start >= 0:
        words.add(text[start:])
    return frozenset(words)


def record_elements(record: dict[str, str | None], fields: tuple[str, ...]) -> frozenset[tuple[str, str]]:
    """Build the element set of a record: (field, word) pairs over match fields.

    The same word in different fields is a different element; missing fields
    are treated like null and produce no words.
    """
    elements: set[tuple[str, str]] = set()
    for field in fields:
        for word in tokenize(record.get(field)):
            elements.add((field, word))
    return frozenset(elements)
