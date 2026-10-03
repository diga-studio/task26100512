"""Randomized differential tests against the exhaustive reference."""

from __future__ import annotations

import random

from fuzzy_dedup import DedupMatcher

from .reference import brute_force_pairs, pairs_with_one_end_in_batch

FIELDS = ("name", "desc", "tag")

# Pool with heavy hitters plus unicode-normalization traps.
POOL = [
    "apple", "banana", "cherry", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango",
    "café", "café", "ﬁle", "ＦＵＬＬ", "ｆｕｌｌ", "straße", "STRASSE",
    "①", "Ⅷ", "Å", "å",
]


def _random_row(rng: random.Random, rid: int) -> dict:
    row: dict = {"id": f"r{rid}"}
    for field in FIELDS:
        roll = rng.random()
        if roll < 0.10:
            continue  # missing
        if roll < 0.18:
            row[field] = None
            continue
        if roll < 0.24:
            row[field] = ""
            continue
        count = rng.randint(1, 10)
        tokens = [rng.choice(POOL) for _ in range(count)]
        # sprinkle duplicates and punctuation/noise inside the raw string
        parts = []
        for tok in tokens:
            parts.append(tok)
            if rng.random() < 0.25:
                parts.append(tok)  # duplicate word inside the same field
            if rng.random() < 0.15:
                parts.append(rng.choice([",", "!", "...", "-", "_", "@"]))
        row[field] = " ".join(parts)
    return row


def _make_records(seed: int, n: int) -> list[dict]:
    rng = random.Random(seed)
    return [_random_row(rng, i) for i in range(n)]


def _split(records: list[dict], sizes: list[int]) -> list[list[dict]]:
    batches, pos = [], 0
    for size in sizes:
        batches.append(records[pos:pos + size])
        pos += size
    assert pos == len(records)
    return batches


def _run_and_check(batches: list[list[dict]]) -> set[tuple[str, str]]:
    """Import batches one by one; after each import, check find_pairs for
    every already-imported batch against the exhaustive reference."""
    matcher = DedupMatcher(FIELDS)
    all_rows: list[dict] = []
    batch_of: dict[str, str] = {}
    batch_ids: list[str] = []
    for idx, batch in enumerate(batches):
        batch_id = f"batch{idx}"
        matcher.import_batch(batch_id, batch)
        all_rows.extend(batch)
        for row in batch:
            batch_of[row["id"]] = batch_id
        batch_ids.append(batch_id)
        for existing in batch_ids:
            got = {tuple(sorted(p)) for p in matcher.find_pairs(existing)}
            want = pairs_with_one_end_in_batch(FIELDS, all_rows, batch_of, existing)
            assert got == want, f"after importing {batch_id}, query {existing} differs"
    # final full pair set across all records
    full: set[tuple[str, str]] = set()
    for existing in batch_ids:
        full |= {tuple(sorted(p)) for p in matcher.find_pairs(existing)}
    assert full == brute_force_pairs(FIELDS, all_rows)
    return full


def test_random_batches_match_reference() -> None:
    for seed in (1, 2, 3):
        records = _make_records(seed, 90)
        _run_and_check(_split(records, [30, 20, 25, 15]))


def test_import_order_and_splitting_do_not_matter() -> None:
    records = _make_records(7, 80)
    reference = _run_and_check(_split(records, [20, 20, 20, 20]))

    reversed_records = list(reversed(records))
    assert _run_and_check(_split(reversed_records, [20, 20, 20, 20])) == reference

    rng = random.Random(99)
    shuffled = records[:]
    rng.shuffle(shuffled)
    assert _run_and_check(_split(shuffled, [5, 40, 13, 22])) == reference

    assert _run_and_check(_split(records, [80])) == reference
    assert _run_and_check(_split(records, [1] * 80)) == reference


def test_empty_sets_never_match() -> None:
    matcher = DedupMatcher(FIELDS)
    matcher.import_batch("b1", [
        {"id": "e1"},                                  # all fields missing
        {"id": "e2", "name": None, "desc": "", "tag": None},
        {"id": "e3", "name": "!!! --- ___", "desc": "@#"},
        {"id": "e4", "name": "apple"},
    ])
    matcher.import_batch("b2", [
        {"id": "e5", "name": "", "desc": None},
        {"id": "e6", "name": "apple"},
    ])
    for batch in ("b1", "b2"):
        pairs = matcher.find_pairs(batch)
        assert all("e1" not in p and "e2" not in p and "e3" not in p and "e5" not in p
                   for p in pairs)
    assert ("e4", "e6") in [tuple(sorted(p)) for p in matcher.find_pairs("b1")]


def test_duplicate_words_and_cross_field_elements() -> None:
    matcher = DedupMatcher(("f1", "f2"))
    matcher.import_batch("b1", [
        {"id": "dup", "f1": "apple apple apple apple"},   # set: {apple}
        {"id": "single", "f1": "apple"},                  # set: {apple}
        {"id": "otherfield", "f2": "apple"},              # set: {(f2,apple)}
        {"id": "both", "f1": "apple", "f2": "apple"},     # 2 elements
    ])
    pairs = {tuple(sorted(p)) for p in matcher.find_pairs("b1")}
    assert ("dup", "single") in pairs          # identical one-element sets
    assert ("dup", "otherfield") not in pairs  # same word, different field
    assert ("otherfield", "single") not in pairs
    assert ("both", "single") not in pairs     # inter 1, union 2 -> 50%


def test_unicode_normalization() -> None:
    matcher = DedupMatcher(("f",))
    matcher.import_batch("b1", [
        {"id": "u1", "f": "ＦＵＬＬｗｉｄｔｈ"},   # fullwidth
        {"id": "u2", "f": "fullwidth"},
        {"id": "u3", "f": "café"},             # e + combining accent
        {"id": "u4", "f": "Café"},
        {"id": "u5", "f": "ﬁle"},                 # fi ligature
        {"id": "u6", "f": "file"},
        {"id": "u7", "f": "straße"},
        {"id": "u8", "f": "STRASSE"},
    ])
    pairs = {tuple(sorted(p)) for p in matcher.find_pairs("b1")}
    assert ("u1", "u2") in pairs
    assert ("u3", "u4") in pairs
    assert ("u5", "u6") in pairs
    assert ("u7", "u8") in pairs


def test_exact_threshold_boundary() -> None:
    matcher = DedupMatcher(("f",))
    rows = []
    # inter 19, union 20 -> 1900 >= 1900: match (exactly 95%)
    rows.append({"id": "a20", "f": " ".join(f"w{i}" for i in range(20))})
    rows.append({"id": "b19", "f": " ".join(f"w{i}" for i in range(19))})
    # inter 18, union 20 -> 1800 < 1900: no match
    rows.append({"id": "c18", "f": " ".join(f"w{i}" for i in range(18))})
    # inter 57, union 60 -> 5700 >= 5700: match
    rows.append({"id": "d60", "f": " ".join(f"x{i}" for i in range(60))})
    rows.append({"id": "e57", "f": " ".join(f"x{i}" for i in range(57))})
    # inter 56, union 60 -> 5600 < 5700: no match
    rows.append({"id": "f56", "f": " ".join(f"x{i}" for i in range(56))})
    matcher.import_batch("b1", rows)
    pairs = {tuple(sorted(p)) for p in matcher.find_pairs("b1")}
    assert ("a20", "b19") in pairs
    assert ("a20", "c18") not in pairs
    assert ("d60", "e57") in pairs
    assert ("d60", "f56") not in pairs

def test_placeholder() -> None:
    pass
