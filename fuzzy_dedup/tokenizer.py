"""Tokenization per the matching spec.

A field value is normalized with NFKC, then casefolded, then split into
maximal runs of characters for which str.isalnum() is true.  None, missing
and empty values produce no tokens.
"""

from __future__ import annotations

import unicodedata


def tokenize_value(value: object) -> list[str]:
    """Return the token list for a single raw field value."""
    if not isinstance(value, str) or not value:
        return []
    text = unicodedata.normalize("NFKC", value).casefold()
    tokens: list[str] = []
    buf: list[str] = []
    for ch in text:
        if ch.isalnum():
            buf.append(ch)
        elif buf:
            tokens.append("".join(buf))
            buf.clear()
    if buf:
        tokens.append("".join(buf))
    return tokens


def record_elements(fields: tuple[str, ...], row: dict) -> set[tuple[int, str]]:
    """Return the element set of a record: (field_index, token) pairs.

    Duplicate tokens inside one field are counted once; the same token in
    different fields yields distinct elements.
    """
    elements: set[tuple[int, str]] = set()
    for field_index, field in enumerate(fields):
        value = row.get(field)
        if value is None:
            continue
        for token in set(tokenize_value(value)):
            elements.add((field_index, token))
    return elements
