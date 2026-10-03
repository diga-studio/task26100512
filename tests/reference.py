"""Independent exhaustive reference implementation used by the tests."""

from __future__ import annotations

from fuzzy_dedup.tokenizer import record_elements


def matches(set_a: set, set_b: set) -> bool:
    """True iff 100 * |A and B| >= 95 * |A or B| (empty sets never match)."""
    union = len(set_a) + len(set_b) - len(set_a & set_b)
    if union == 0:
        return False
    return 100 * len(set_a & set_b) >= 95 * union


def brute_force_pairs(fields: tuple[str, ...], rows: list[dict]) -> set[tuple[str, str]]:
    """All matching pairs among the given rows, computed by enumeration."""
    sets = [(row["id"], record_elements(fields, row)) for row in rows]
    pairs: set[tuple[str, str]] = set()
    for i in range(len(sets)):
        id_a, set_a = sets[i]
        for j in range(i + 1, len(sets)):
            id_b, set_b = sets[j]
            if matches(set_a, set_b):
                pairs.add((id_a, id_b) if id_a < id_b else (id_b, id_a))
    return pairs


def pairs_with_one_end_in_batch(
    fields: tuple[str, ...], rows: list[dict], batch_of: dict[str, str], batch_id: str
) -> set[tuple[str, str]]:
    """Brute-force find_pairs: pairs with at least one endpoint in batch_id."""
    sets = [(row["id"], record_elements(fields, row)) for row in rows]
    pairs: set[tuple[str, str]] = set()
    for i in range(len(sets)):
        id_a, set_a = sets[i]
        in_a = batch_of[id_a] == batch_id
        for j in range(i + 1, len(sets)):
            id_b, set_b = sets[j]
            if not in_a and batch_of[id_b] != batch_id:
                continue
            if matches(set_a, set_b):
                pairs.add((id_a, id_b) if id_a < id_b else (id_b, id_a))
    return pairs
