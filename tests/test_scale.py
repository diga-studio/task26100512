"""Scale test: 100k records, 20 batches, exactly 1000 expected pairs."""

from __future__ import annotations

import random
import resource
import sys
import time

from fuzzy_dedup import DedupMatcher

N_BASE = 99_000
N_VARIANT = 1_000
N_COMMON = 56
N_PRIVATE = 8
N_BATCHES = 20
CPU_LIMIT_S = 60.0
RSS_LIMIT_BYTES = 512 * 1024 * 1024

COMMON = [f"common{k}" for k in range(N_COMMON)]


def _peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KiB
    return rss if sys.platform == "darwin" else rss * 1024


def _generate_rows() -> tuple[list[dict], dict[str, str]]:
    """Return (rows, variant->base mapping)."""
    rows: list[dict] = []
    variant_of: dict[str, str] = {}
    for i in range(N_BASE):
        private = [f"p{i}x{k}" for k in range(N_PRIVATE)]
        rows.append({"id": f"base{i}", "text": " ".join(COMMON + private)})
    # each variant derives from a distinct base, replacing one private token
    for j in range(N_VARIANT):
        base_idx = j * (N_BASE // N_VARIANT)
        private = [f"p{base_idx}x{k}" for k in range(N_PRIVATE)]
        private[0] = f"v{j}fresh"
        variant_id = f"var{j}"
        variant_of[variant_id] = f"base{base_idx}"
        rows.append({"id": variant_id, "text": " ".join(COMMON + private)})
    rng = random.Random(20261005)
    rng.shuffle(rows)
    return rows, variant_of


def test_scale_100k() -> None:
    start_cpu = time.process_time()
    rows, variant_of = _generate_rows()
    assert len(rows) == N_BASE + N_VARIANT

    matcher = DedupMatcher(("text",))
    batch_size = len(rows) // N_BATCHES
    all_pairs: set[tuple[str, str]] = set()
    for b in range(N_BATCHES):
        batch = rows[b * batch_size:(b + 1) * batch_size]
        for i in range(b * batch_size, (b + 1) * batch_size):
            rows[i] = None  # release processed rows to keep peak RSS down
        batch_id = f"batch{b}"
        matcher.import_batch(batch_id, batch)
        for pair in matcher.find_pairs(batch_id):
            all_pairs.add(tuple(sorted(pair)))
        del batch
    del rows

    cpu = time.process_time() - start_cpu
    rss = _peak_rss_bytes()
    print(f"\nscale test: cpu={cpu:.2f}s (limit {CPU_LIMIT_S:.0f}s), "
          f"peak_rss={rss / 1024 / 1024:.1f} MiB (limit 512 MiB), "
          f"pairs={len(all_pairs)}")

    expected = {tuple(sorted((vid, bid))) for vid, bid in variant_of.items()}
    assert all_pairs == expected
    assert len(all_pairs) == N_VARIANT
    assert cpu <= CPU_LIMIT_S, f"cpu {cpu:.1f}s exceeds {CPU_LIMIT_S}s"
    assert rss <= RSS_LIMIT_BYTES, f"rss {rss} exceeds 512 MiB"
