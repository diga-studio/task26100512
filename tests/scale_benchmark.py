"""Scale benchmark, runnable standalone or via tests/test_scale.py.

100,000 records x 64 words each; 56 words shared by all records. 99,000 base
records each carry 8 globally unique private words; 1,000 variants are each
derived from a distinct base by replacing exactly one private word with a new
unique word. Imported in 20 batches of 5,000; each batch is queried right
after its import. Exactly 1,000 matching pairs must be found.

Prints total CPU time (generation + imports + queries) and peak RSS.
"""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fuzzy_dedup import Matcher  # noqa: E402

SHARED = 56
PRIVATE = 8
BASES = 99_000
VARIANTS = 1_000
BATCHES = 20
TIME_LIMIT_S = 60.0
RSS_LIMIT_BYTES = 512 * 1024 * 1024


SHARED_WORDS = [f"shared{w}" for w in range(SHARED)]


def make_row(spec: tuple) -> dict:
    """Materialize one record. Specs stay tiny; rows are built lazily per
    batch so the benchmark itself does not hold all texts at once."""
    kind, num = spec[0], spec[1]
    if kind == "base":
        private = [f"p{num}w{j}" for j in range(PRIVATE)]
        return {"id": f"base{num}", "text": " ".join(SHARED_WORDS + private)}
    base = num * (BASES // VARIANTS)
    private = [f"q{num}new" if j == 0 else f"p{base}w{j}" for j in range(PRIVATE)]
    return {"id": f"var{num}of{base}", "text": " ".join(SHARED_WORDS + private)}


def build_specs() -> list[tuple]:
    """One lightweight descriptor per record: 99,000 bases + 1,000 variants,
    each variant derived from a distinct, evenly spread base that replaces
    exactly one private word with a brand-new unique word."""
    specs = [("base", i) for i in range(BASES)]
    specs += [("var", k) for k in range(VARIANTS)]
    # Interleave deterministically so variants land in different batches.
    specs.sort(key=lambda spec: hash((spec, "spread")) & 0x7FFFFFFF)
    return specs


def main() -> int:
    cpu_start = time.process_time()
    wall_start = time.perf_counter()

    specs = build_specs()
    matcher = Matcher(fields=["text"])
    all_pairs: set[tuple[str, str]] = set()
    per_batch = len(specs) // BATCHES
    for batch_num in range(BATCHES):
        batch_id = f"batch{batch_num}"
        rows = [make_row(s) for s in specs[batch_num * per_batch:(batch_num + 1) * per_batch]]
        matcher.import_batch(batch_id, rows)
        for a, b in matcher.find_pairs(batch_id):
            all_pairs.add((a, b) if a < b else (b, a))

    cpu = time.process_time() - cpu_start
    wall = time.perf_counter() - wall_start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if sys.platform == "darwin" else rss * 1024

    print(f"pairs found : {len(all_pairs)}")
    print(f"CPU time    : {cpu:.2f} s (limit {TIME_LIMIT_S:.0f} s)")
    print(f"wall time   : {wall:.2f} s")
    print(f"peak RSS    : {rss_bytes / 1024 / 1024:.1f} MiB (limit 512 MiB)")

    assert len(all_pairs) == VARIANTS, f"expected {VARIANTS} pairs, got {len(all_pairs)}"
    for a, b in all_pairs:
        base_id, var_id = (a, b) if a.startswith("base") else (b, a)
        assert var_id == f"var{int(base_id[4:]) // (BASES // VARIANTS)}of{base_id[4:]}", (a, b)
    assert cpu <= TIME_LIMIT_S, f"CPU time {cpu:.1f}s exceeds {TIME_LIMIT_S}s"
    assert rss_bytes <= RSS_LIMIT_BYTES, f"RSS {rss_bytes} exceeds 512 MiB"
    print("SCALE BENCHMARK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
