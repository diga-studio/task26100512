"""Exact cross-batch fuzzy dedup matcher (Jaccard >= 0.95 on token sets)."""

from __future__ import annotations

from array import array

from .tokenizer import tokenize_value

_MASK = (1 << 64) - 1

# Threshold: report iff 100 * |A n B| >= 95 * |A u B|, i.e. 39*i >= 19*(a+b).
_NUM = 95
_DEN = 100


def _mix64(x: int) -> int:
    """splitmix64 finalizer: deterministic 64-bit mixing."""
    x &= _MASK
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & _MASK
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & _MASK
    return x ^ (x >> 31)


_SEED_H1 = 0x9E3779B97F4A7C15
_SEED_H2 = 0xC2B2AE3D27D4EB4F
_SEED_G1 = 0x165667B19E3779F9
_SEED_G2 = 0xD6E8FEB86659FD93


def _sig_key(level: int, part: int, bucket: int, size: int, content: int) -> int:
    meta = _mix64((size << 32) ^ (level << 16) ^ (part << 8) ^ bucket)
    return _mix64((content ^ meta) & _MASK)


def _size_range(a: int) -> tuple[int, int]:
    """Feasible partner sizes b for a record of size a (19b<=20a, 19a<=20b)."""
    return (19 * a + 19) // 20, 20 * a // 19


def _mismatches(a: int, b: int) -> int:
    """Max symmetric-difference size for a matching pair of sizes a, b."""
    i_req = -(-19 * (a + b) // 39)  # ceil: 39*i >= 19*(a+b)
    return a + b - 2 * i_req


def _levels_for_size(a: int) -> tuple[int, ...]:
    """The levels a record of size a must index.

    Only levels d(a,b)+1 for feasible b are ever probed, so exactly those
    are indexed (a small set because d stays near a/39..a/19).
    """
    b_min, b_max = _size_range(a)
    return tuple(sorted({_mismatches(a, b) + 1 for b in range(b_min, b_max + 1)
                         if _mismatches(a, b) >= 0}))


def _intersect_count(xs: array, ys: array) -> int:
    i = j = n = 0
    lx, ly = len(xs), len(ys)
    while i < lx and j < ly:
        x = xs[i]
        y = ys[j]
        if x == y:
            n += 1
            i += 1
            j += 1
        elif x < y:
            i += 1
        else:
            j += 1
    return n


class DedupMatcher:
    """Matcher over records with globally unique string ids.

    The participating fields are fixed at construction time.  Records are
    only ever added; find_pairs reports every pair (at 95% Jaccard on the
    token-set definition) with at least one endpoint in the queried batch.
    """

    def __init__(self, fields: list[str] | tuple[str, ...]) -> None:
        if not fields:
            raise ValueError("at least one field is required")
        self.fields: tuple[str, ...] = tuple(fields)
        # one token->id map per field: the same word in different fields is a
        # different element, exactly as the spec requires
        self._token_ids: list[dict[str, int]] = [{} for _ in self.fields]
        self._h1: array = array("Q")
        self._h2: array = array("Q")
        self._g1: array = array("Q")
        self._g2: array = array("Q")
        self._records: list[array] = []
        self._record_ids: list[str] = []
        self._rid_by_id: dict[str, int] = {}
        self._batches: dict[str, list[int]] = {}
        # signature key -> rid (singleton) or list[rid]; collisions only ever
        # add false candidates, never lose true ones
        self._index: dict[int, int | list[int]] = {}
        self._levels_cache: dict[int, tuple[int, ...]] = {}

    def _levels(self, size: int) -> tuple[int, ...]:
        levels = self._levels_cache.get(size)
        if levels is None:
            levels = _levels_for_size(size)
            self._levels_cache[size] = levels
        return levels

    def _intern(self, row: dict) -> array:
        h1, h2, g1, g2 = self._h1, self._h2, self._g1, self._g2
        out: set[int] = set()
        for field_index, field in enumerate(self.fields):
            value = row.get(field)
            if not isinstance(value, str) or not value:
                continue
            token_ids = self._token_ids[field_index]
            for token in set(tokenize_value(value)):
                tid = token_ids.get(token)
                if tid is None:
                    tid = len(self._h1)
                    token_ids[token] = tid
                    h1.append(_mix64(tid + _SEED_H1))
                    h2.append(_mix64(tid + _SEED_H2))
                    g1.append(_mix64(tid + _SEED_G1))
                    g2.append(_mix64(tid + _SEED_G2))
                out.add(tid)
        return array("I", sorted(out))

    def _add_to_index(self, rid: int, tokens: array) -> None:
        size = len(tokens)
        if size == 0:
            return
        h1, h2, g1, g2 = self._h1, self._h2, self._g1, self._g2
        index = self._index
        for level in self._levels(size):
            for part in (0, 1):
                hs = h1 if part == 0 else h2
                gs = g1 if part == 0 else g2
                sums = [0] * level
                for tid in tokens:
                    bucket = hs[tid] % level
                    sums[bucket] = (sums[bucket] + gs[tid]) & _MASK
                for bucket, content in enumerate(sums):
                    key = _sig_key(level, part, bucket, size, content)
                    val = index.get(key)
                    if val is None:
                        index[key] = rid
                    elif type(val) is list:
                        val.append(rid)
                    else:
                        index[key] = [val, rid]

    def import_batch(self, batch_id: str, rows: list[dict]) -> None:
        """Import a batch of up to 5000 records."""
        if batch_id in self._batches:
            raise ValueError(f"duplicate batch id: {batch_id!r}")
        rids: list[int] = []
        self._batches[batch_id] = rids
        for row in rows:
            record_id = row["id"]
            if record_id in self._rid_by_id:
                raise ValueError(f"duplicate record id: {record_id!r}")
            rid = len(self._records)
            tokens = self._intern(row)
            self._records.append(tokens)
            self._record_ids.append(record_id)
            self._rid_by_id[record_id] = rid
            rids.append(rid)
            self._add_to_index(rid, tokens)

    def _candidates(self, rid: int) -> set[int]:
        qtokens = self._records[rid]
        b = len(qtokens)
        if b == 0:
            return set()
        # feasible sizes a satisfy 19*a <= 20*b and 19*b <= 20*a
        a_min = (19 * b + 19) // 20
        a_max = 20 * b // 19
        # bucket content sums per (level, part), computed lazily
        sums_cache: dict[int, tuple[list[int], list[int]]] = {}
        candidates: set[int] = set()
        index = self._index
        h1, h2, g1, g2 = self._h1, self._h2, self._g1, self._g2
        for a in range(a_min, a_max + 1):
            i_req = -(-19 * (a + b) // 39)
            d = a + b - 2 * i_req
            if d < 0:
                continue
            level = d + 1
            cached = sums_cache.get(level)
            if cached is None:
                sums0 = [0] * level
                sums1 = [0] * level
                for tid in qtokens:
                    sums0[h1[tid] % level] = (sums0[h1[tid] % level] + g1[tid]) & _MASK
                    sums1[h2[tid] % level] = (sums1[h2[tid] % level] + g2[tid]) & _MASK
                cached = (sums0, sums1)
                sums_cache[level] = cached
            per_part: list[set[int] | None] = []
            for part in (0, 1):
                sums = cached[part]
                found: set[int] | None = None
                for bucket, content in enumerate(sums):
                    val = index.get(_sig_key(level, part, bucket, a, content))
                    if val is None:
                        continue
                    if found is None:
                        found = {val} if type(val) is int else set(val)
                    elif type(val) is int:
                        found.add(val)
                    else:
                        found.update(val)
                if not found:
                    per_part = None
                    break
                per_part.append(found)
            if per_part is None:
                continue
            both = per_part[0] & per_part[1]
            candidates.update(both)
        candidates.discard(rid)
        return candidates

    def find_pairs(self, batch_id: str) -> list[tuple[str, str]]:
        """All matching pairs with at least one endpoint in batch_id."""
        rids = self._batches[batch_id]
        records = self._records
        pairs: set[tuple[int, int]] = set()
        for rid in rids:
            qtokens = records[rid]
            b = len(qtokens)
            if b == 0:
                continue
            for cand in self._candidates(rid):
                ctokens = records[cand]
                inter = _intersect_count(qtokens, ctokens)
                union = b + len(ctokens) - inter
                if _DEN * inter >= _NUM * union:
                    pairs.add((rid, cand) if rid < cand else (cand, rid))
        record_ids = self._record_ids
        return sorted((record_ids[r1], record_ids[r2]) for r1, r2 in pairs)
