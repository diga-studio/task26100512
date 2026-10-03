"""Independent brute-force reference implementation.

Recomputes, from the raw record definitions, the exact set of matching pairs
with at least one endpoint in the queried batch. O(n^2) on purpose: it shares
no code with the library under test.
"""

from __future__ import annotations

import unicodedata

THRESHOLD_NUM = 95
THRESHOLD_DEN = 100


def brute_elements(record: dict, fields: tuple[str, ...]) -> frozenset[tuple[str, str]]:
    elements: set[tuple[str, str]] = set()
    for field in fields:
        value = record.get(field)
        if not value:
            continue
        text = unicodedata.normalize("NFKC", value).casefold()
        word: list[str] = []
        for char in text:
            if char.isalnum():
                word.append(char)
            elif word:
                elements.add((field, "".join(word)))
                word = []
        if word:
            elements.add((field, "".join(word)))
    return frozenset(elements)


def brute_matches(set_a: frozenset, set_b: frozenset) -> bool:
    if not set_a or not set_b:
        return False
    inter = len(set_a & set_b)
    union = len(set_a) + len(set_b) - inter
    return inter * THRESHOLD_DEN >= THRESHOLD_NUM * union


def brute_find_pairs(
    records: list[tuple[str, dict]],  # (batch_id, row) in import order
    fields: tuple[str, ...],
    batch_id: str,
) -> set[tuple[str, str]]:
    indexed = [(row["id"], brute_elements(row, fields), bid) for bid, row in records]
    pairs: set[tuple[str, str]] = set()
    for i in range(len(indexed)):
        id_a, set_a, batch_a = indexed[i]
        for j in range(i + 1, len(indexed)):
            id_b, set_b, batch_b = indexed[j]
            if batch_id not in (batch_a, batch_b):
                continue
            if brute_matches(set_a, set_b):
                pairs.add((id_a, id_b) if id_a < id_b else (id_b, id_a))
    return pairs
