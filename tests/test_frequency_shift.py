"""Later batches may drastically change word frequencies. Matches must still
be found, including pairs whose overlap consists ONLY of words that have
become high-frequency by the time the second record arrives."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fuzzy_dedup import Matcher  # noqa: E402

FIELDS = ("text",)


def test_old_batch_finds_match_after_frequency_shift():
    matcher = Matcher(fields=FIELDS)

    # Batch 1: record A is built entirely from words that are RARE right now.
    hot = [f"hot{i}" for i in range(20)]
    matcher.import_batch("b1", [{"id": "A", "text": " ".join(hot)}])

    # Batch 2..4: hundreds of records turn hot0..hot19 into high-frequency
    # words (each filler also carries its own unique word so fillers never
    # match each other or A).
    for batch_num in range(3):
        rows = [
            {"id": f"filler{batch_num}_{i}", "text": " ".join(hot[:10] + [f"uniq{batch_num}_{i}"])}
            for i in range(100)
        ]
        matcher.import_batch(f"fill{batch_num}", rows)

    # Batch 5: record B overlaps A ONLY through the now-high-frequency words:
    # B = hot0..hot18 (a 19-element subset of A) -> 19/20 = exactly 95% with A.
    matcher.import_batch("b5", [{"id": "B", "text": " ".join(hot[:19])}])

    # The old batch b1 must now report (A, B).
    assert sorted(matcher.find_pairs("b1")) == [("A", "B")]
    assert sorted(matcher.find_pairs("b5")) == [("A", "B")]
    # Fillers share 10 of 12 union elements among themselves: 83% -> no pairs.
    for batch_num in range(3):
        assert matcher.find_pairs(f"fill{batch_num}") == []


def test_rare_word_pair_also_survives():
    """Complementary case: a pair sharing a rare word must obviously match,
    included to show the high-frequency test above is the discriminating one."""
    matcher = Matcher(fields=FIELDS)
    rare = [f"rare{i}" for i in range(20)]
    matcher.import_batch("b1", [{"id": "A", "text": " ".join(rare)}])
    matcher.import_batch("b2", [{"id": "B", "text": " ".join(rare[:19])}])
    assert sorted(matcher.find_pairs("b1")) == [("A", "B")]
