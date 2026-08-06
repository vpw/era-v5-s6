"""Vocabulary projection for the toy model.

The frozen tokenizer has 68,096 entries. Shards store those real ids and every hash in the
system is computed over them -- that never changes, because the tokenizer is a Session 2
contract and a shard is only meaningful under the vocabulary that built it.

But a 68k output layer would dominate a demo whose point is the data system: the logit
tensor alone would be 139 MB per microbatch, and the embedding table would be six times
the size of the rest of the model. So the *model* trains on a compact projection of the
vocabulary -- the most frequent `size` token ids across the admitted shards, with
everything else folded onto `<unk>`.

This is a toy-model concession and it is deliberately kept at arm's length from the data
path:

  * shards, ledger events, batch hashes and replay comparisons all use real tokenizer ids;
  * the projection is derived from the admitted shards by frequency, so it is
    reproducible, and its hash is recorded in the artifacts;
  * nothing in the projection can change which documents were selected, packed or
    consumed -- it only decides what the model's output layer is wide enough to name.

The coverage figure it reports (what share of corpus token occurrences survive the
projection rather than becoming `<unk>`) is written into the performance report, so the
cost of the concession is visible rather than assumed.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .hashing import sha256_bytes, short, tagged
from .manifest import ShardRegistry


class VocabProjection:
    def __init__(self, ids: list[int], unk_id: int, source_vocab_size: int):
        self.ids = list(ids)
        self.unk_id = unk_id
        self.source_vocab_size = source_vocab_size
        self.size = len(self.ids)

        # Dense lookup: real tokenizer id -> compact id. Everything unmapped goes to the
        # compact slot that holds <unk>.
        self.unk_slot = self.ids.index(unk_id)
        self.forward = np.full(source_vocab_size, self.unk_slot, dtype=np.int32)
        for compact, real in enumerate(self.ids):
            self.forward[real] = compact
        self.backward = np.asarray(self.ids, dtype=np.int32)

    def project(self, token_ids: np.ndarray) -> np.ndarray:
        return self.forward[np.clip(token_ids, 0, self.source_vocab_size - 1)]

    def restore(self, compact_ids: np.ndarray) -> np.ndarray:
        return self.backward[compact_ids]

    @property
    def hash(self) -> str:
        return tagged("vocab", sha256_bytes(self.backward.tobytes()))

    def descriptor(self, coverage: float | None = None) -> dict:
        return {
            "projected_vocab_size": self.size,
            "source_vocab_size": self.source_vocab_size,
            "vocab_projection_hash": self.hash,
            "unk_compact_id": self.unk_slot,
            "token_coverage": coverage,
            "_doc": "The model's output layer is this wide. Shards, ledger events and "
                    "batch hashes always use the full 68,096-entry tokenizer ids.",
        }

    def write(self, path: Path, coverage: float | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.descriptor(coverage)
        payload["token_ids"] = self.ids
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True)
        return path


def build_projection(
    registry: ShardRegistry, size: int, special_ids: list[int], source_vocab_size: int, unk_id: int
) -> tuple[VocabProjection, float]:
    """Count token occurrences across admitted shards and keep the most frequent `size`.

    Ties are broken by token id so the result does not depend on counter ordering.
    """
    counts: Counter[int] = Counter()
    for entry in registry.admitted:
        tokens = np.asarray(entry.shard.tokens())
        values, occurrences = np.unique(tokens, return_counts=True)
        counts.update(dict(zip(values.tolist(), occurrences.tolist())))

    reserved = sorted(set(special_ids) | {unk_id})
    budget = max(size - len(reserved), 0)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    kept = [token for token, _ in ranked if token not in reserved][:budget]

    ids = sorted(set(reserved) | set(kept))
    projection = VocabProjection(ids, unk_id=unk_id, source_vocab_size=source_vocab_size)

    total = sum(counts.values())
    covered = sum(count for token, count in counts.items() if token in set(ids))
    coverage = round(covered / total, 4) if total else 0.0
    return projection, coverage
