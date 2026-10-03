"""Exact Jaccard (>= 0.95) matching across sequential import batches.

Candidate generation uses prefix filtering over an inverted index:

- Elements of every record are ordered by their document frequency at the
  moment the record is imported (ties broken by element id). This order is
  only a performance heuristic; correctness never depends on it.
- A record with n elements indexes its first p(n) = n - ceil(0.95*n) + 1
  elements (its "prefix") into the inverted index.
- A new record probes the index with ALL of its elements and verifies every
  candidate exactly.

Why no match is missed: for a matching pair (x, y) with sizes n <= m,
Jaccard >= 0.95 implies m <= n/0.95, hence the overlap
|x∩y| >= 0.95*(n+m)/1.95 >= 0.95*m > ceil(0.95*m) - 1 = |suffix(y)|.
So at least one shared element lies in y's indexed prefix, and x probes with
all of its elements, therefore y is always found as a candidate. The frozen
prefix of y stays valid forever, regardless of how word frequencies, import
order, or batching change later.
"""

from __future__ import annotations

from array import array

from fuzzy_dedup.tokenizer import tokenize

THRESHOLD_NUM = 95
THRESHOLD_DEN = 100
MAX_BATCH_SIZE = 5000


def _prefix_len(size: int) -> int:
    """Number of leading elements (in frequency order) that get indexed."""
    return size - (THRESHOLD_NUM * size + THRESHOLD_DEN - 1) // THRESHOLD_DEN + 1


def _intersection_size(a: array, b: array) -> int:
    """Size of the intersection of two sorted integer arrays."""
    i = j = count = 0
    len_a, len_b = len(a), len(b)
    while i < len_a and j < len_b:
        if a[i] == b[j]:
            count += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return count


class Matcher:
    """Incremental exact fuzzy-dedup matcher.

    Created with a fixed set of match fields. Records are only ever appended
    via import_batch(); find_pairs() reports all matching unordered pairs with
    at least one endpoint in the requested batch.
    """

    def __init__(self, fields: list[str] | tuple[str, ...]) -> None:
        if not fields:
            raise ValueError("at least one match field is required")
        self._fields = tuple(fields)
        # Per-field interning of words to small integer element ids.
        self._word_ids: dict[str, dict[str, int]] = {f: {} for f in self._fields}
        self._doc_freq: array = array("i")  # element id -> records containing it
        self._postings: dict[int, list[int]] = {}  # prefix element id -> record indexes
        self._record_sets: list[array] = []  # record index -> sorted element ids
        self._record_ids: list[str] = []  # record index -> global record id
        self._record_batch: list[str] = []  # record index -> batch id
        self._batch_pairs: dict[str, set[tuple[int, int]]] = {}

    @property
    def fields(self) -> tuple[str, ...]:
        return self._fields

    def _elements_of(self, row: dict) -> array:
        """Sorted unique element ids for a row, interning new words."""
        element_ids: set[int] = set()
        for field in self._fields:
            value = row.get(field)
            if not value:
                continue
            word_ids = self._word_ids[field]
            for word in tokenize(value):
                eid = word_ids.get(word)
                if eid is None:
                    eid = len(self._doc_freq)
                    word_ids[word] = eid
                    self._doc_freq.append(0)
                element_ids.add(eid)
        return array("i", sorted(element_ids))

    def import_batch(self, batch_id: str, rows: list[dict]) -> None:
        if len(rows) > MAX_BATCH_SIZE:
            raise ValueError(f"batch size {len(rows)} exceeds {MAX_BATCH_SIZE}")
        if batch_id in self._batch_pairs:
            raise ValueError(f"duplicate batch id: {batch_id!r}")
        self._batch_pairs[batch_id] = set()
        for row in rows:
            self._add_record(batch_id, row)

    def _add_record(self, batch_id: str, row: dict) -> None:
        elements = self._elements_of(row)
        index = len(self._record_sets)
        self._record_sets.append(elements)
        self._record_ids.append(row["id"])
        self._record_batch.append(batch_id)

        size = len(elements)
        if size == 0:
            return  # empty sets never match anything

        # Probe: every record sharing an indexed prefix element is a candidate.
        candidates: set[int] = set()
        postings = self._postings
        for eid in elements:
            lst = postings.get(eid)
            if lst:
                candidates.update(lst)

        # Exact verification; collect matching pairs.
        for other in candidates:
            other_set = self._record_sets[other]
            inter = _intersection_size(elements, other_set)
            union = size + len(other_set) - inter
            if inter * THRESHOLD_DEN >= THRESHOLD_NUM * union:
                pair = (other, index)
                self._batch_pairs[batch_id].add(pair)
                self._batch_pairs[self._record_batch[other]].add(pair)

        # Index this record's prefix under the current frequency order.
        prefix_count = _prefix_len(size)
        doc_freq = self._doc_freq
        ordered = sorted(elements, key=lambda e: (doc_freq[e], e))
        for eid in ordered[:prefix_count]:
            postings.setdefault(eid, []).append(index)

        # Update document frequencies only after this record is processed.
        for eid in elements:
            doc_freq[eid] += 1

    def find_pairs(self, batch_id: str) -> list[tuple[str, str]]:
        pairs = self._batch_pairs.get(batch_id)
        if pairs is None:
            raise KeyError(f"unknown batch id: {batch_id!r}")
        ids = self._record_ids
        return [(ids[a], ids[b]) for a, b in pairs]
