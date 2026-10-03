"""Correctness: library results must equal the brute-force reference exactly,
after every import, under any import order and any batch partitioning."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reference import brute_find_pairs  # noqa: E402

from fuzzy_dedup import Matcher  # noqa: E402

FIELDS = ("f1", "f2")


def make_records(seed: int, count: int) -> list[dict]:
    rng = random.Random(seed)
    vocab = [f"w{i}" for i in range(30)] + ["café", "straße", "ＦＵＬＬ", "naïve"]
    records = []
    for i in range(count):
        row = {"id": f"r{i:03d}"}
        for field in FIELDS:
            roll = rng.random()
            if roll < 0.08:
                continue  # missing field
            if roll < 0.14:
                row[field] = None
                continue
            k = rng.randint(0, 6)
            words = [rng.choice(vocab) for _ in range(k)]
            if rng.random() < 0.3 and words:
                words.append(words[0])  # duplicate word inside a field
            row[field] = " ".join(words)
        records.append(row)
    return records


def run_scenario(records: list[dict], batches: list[list[dict]], order: list[int]):
    """Import batches in the given order; after each import, check find_pairs
    for every known batch against the reference. Returns the final full pair
    set (union over all batches)."""
    matcher = Matcher(fields=FIELDS)
    imported: list[tuple[str, dict]] = []
    batch_ids: list[str] = []
    for step, batch_idx in enumerate(order):
        batch_id = f"batch{batch_idx}"
        matcher.import_batch(batch_id, batches[batch_idx])
        imported.extend((batch_id, row) for row in batches[batch_idx])
        batch_ids.append(batch_id)
        for known in batch_ids:
            got = {tuple(sorted(p)) for p in matcher.find_pairs(known)}
            want = brute_find_pairs(imported, FIELDS, known)
            assert got == want, f"step {step}, query {known}:\nonly got: {got - want}\nmissing: {want - got}"
    full: set[tuple[str, str]] = set()
    for known in batch_ids:
        full |= {tuple(sorted(p)) for p in matcher.find_pairs(known)}
    return full


def partition(records: list[dict], sizes: list[int]) -> list[list[dict]]:
    batches, pos = [], 0
    for size in sizes:
        batches.append(records[pos:pos + size])
        pos += size
    assert pos == len(records)
    return batches


def test_random_matches_reference_after_every_import():
    records = make_records(seed=42, count=120)
    batches = partition(records, [30, 20, 40, 30])
    run_scenario(records, batches, order=[0, 1, 2, 3])


def test_import_order_and_partitioning_invariance():
    records = make_records(seed=7, count=100)
    scenarios = [
        (partition(records, [50, 50]), [0, 1]),
        (partition(records, [50, 50]), [1, 0]),
        (partition(records, [10] * 10), [9, 0, 5, 1, 8, 2, 7, 3, 6, 4]),
        (partition(records, [1] * 100), list(range(100))),
        (partition(records, [100]), [0]),
    ]
    final_sets = [run_scenario(records, batches, order) for batches, order in scenarios]
    for other in final_sets[1:]:
        assert other == final_sets[0]
    # and the final set equals the reference over all records
    all_records = [(f"b{i}", row) for i, row in enumerate(records)]
    want = brute_find_pairs(all_records, FIELDS, "b0")  # batch0 covers only pairs touching r000
    want_all = set()
    for i in range(len(records)):
        want_all |= brute_find_pairs(all_records, FIELDS, f"b{i}")
    assert final_sets[0] == want_all


def test_empty_sets_never_match():
    matcher = Matcher(fields=FIELDS)
    matcher.import_batch("b1", [
        {"id": "empty1"},
        {"id": "empty2", "f1": None, "f2": ""},
        {"id": "punct", "f1": "!!! ---", "f2": " , "},
        {"id": "real", "f1": "hello"},
    ])
    assert matcher.find_pairs("b1") == []


def test_duplicate_words_and_cross_field_semantics():
    matcher = Matcher(fields=FIELDS)
    matcher.import_batch("b1", [
        # "apple" x3 in f1 counts once -> element set {(f1,apple)}
        {"id": "a", "f1": "apple apple apple"},
        # same word in a different field is a different element -> no match
        {"id": "b", "f2": "apple"},
        # identical element set -> match with a
        {"id": "c", "f1": "apple"},
    ])
    assert sorted(matcher.find_pairs("b1")) == [("a", "c")]


def test_unicode_normalization_and_casefold():
    matcher = Matcher(fields=FIELDS)
    matcher.import_batch("b1", [
        {"id": "a", "f1": "ＳＴＲＡＳＳＥ Café"},   # full-width + mixed case
        {"id": "b", "f1": "strasse café"},          # casefold + NFKC equivalent
        {"id": "c", "f1": "ﬁle NAÏVE"},             # ligature fi -> file
        {"id": "d", "f1": "file naïve"},
    ])
    assert sorted(matcher.find_pairs("b1")) == [("a", "b"), ("c", "d")]


def test_exact_threshold_boundary():
    # 19 common / 20 union = exactly 95% -> must match
    common = [f"t{i}" for i in range(19)]
    matcher = Matcher(fields=FIELDS)
    matcher.import_batch("b1", [
        {"id": "a", "f1": " ".join(common + ["onlya"])},        # 20 elements
        {"id": "b", "f1": " ".join(common)},                    # 19 elements, subset
        {"id": "c", "f1": " ".join(common + ["onlyc"])},        # vs a: 19/21 ~ 90.5% -> no
        {"id": "d", "f1": " ".join(common[:18] + ["x", "y"])},  # vs a: 18/22 ~ 81.8% -> no
    ])
    # a-b: 19/20 = exactly 95%; b-c: 19/20 = exactly 95%; everything else below
    assert sorted(matcher.find_pairs("b1")) == [("a", "b"), ("b", "c")]


def test_fields_fixed_at_creation():
    matcher = Matcher(fields=["f1"])
    assert matcher.fields == ("f1",)
    with pytest.raises(ValueError):
        Matcher(fields=[])
