"""Later batches drastically change term frequencies; old matches must survive."""

from __future__ import annotations

from collections import Counter

from fuzzy_dedup import DedupMatcher
from fuzzy_dedup.tokenizer import tokenize_value

FIELDS = ("text",)


def _document_frequency(rows_by_batch: list[list[dict]]) -> Counter:
    df: Counter = Counter()
    for rows in rows_by_batch:
        for row in rows:
            for token in set(tokenize_value(row.get("text"))):
                df[token] += 1
    return df


def test_old_batch_matches_survive_frequency_shift() -> None:
    hot = [f"hot{i}" for i in range(40)]  # will become high-frequency later

    batch1 = [
        # matches a later record purely through soon-to-be-common tokens
        {"id": "old_hot", "text": " ".join(hot)},
        # matches a later record through rare tokens (control case)
        {"id": "old_rare", "text": " ".join(["rarea", "rareb"] + hot[:18])},
    ]
    # batch2 makes every hot* token high-frequency (200 records contain them)
    batch2 = [
        {"id": f"filler{i}", "text": " ".join(hot + [f"uniq{i}"])}
        for i in range(200)
    ]
    # shares 39 hot tokens with old_hot: inter 39, union 41 -> 3900 >= 3895
    batch2.append({"id": "new_hot", "text": " ".join(hot[:39] + ["freshz"])})
    # shares rarea, rareb + 18 hot with old_rare: inter 20, union 20 -> match
    batch2.append({"id": "new_rare", "text": " ".join(["rarea", "rareb"] + hot[:18])})

    matcher = DedupMatcher(FIELDS)
    matcher.import_batch("b1", batch1)
    assert matcher.find_pairs("b1") == []  # nothing matches within batch1
    matcher.import_batch("b2", batch2)

    df = _document_frequency([batch1, batch2])
    assert df["hot0"] >= 200  # the shared tokens really are high-frequency now

    pairs_b1 = {tuple(sorted(p)) for p in matcher.find_pairs("b1")}
    assert ("new_hot", "old_hot") in pairs_b1
    assert ("new_rare", "old_rare") in pairs_b1

    # the old_hot/new_hot pair shares *only* high-frequency tokens
    shared = set(tokenize_value(" ".join(hot))) & set(
        tokenize_value(" ".join(hot[:39] + ["freshz"]))
    )
    assert shared and all(df[t] >= 200 for t in shared)

    # batch2's own view must contain the pair too, and re-querying stays stable
    pairs_b2 = {tuple(sorted(p)) for p in matcher.find_pairs("b2")}
    assert ("new_hot", "old_hot") in pairs_b2
    assert {tuple(sorted(p)) for p in matcher.find_pairs("b1")} == pairs_b1
