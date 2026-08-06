"""Canonical serialisation and hashing.

Every identity in this system -- shard content, tokenizer, batch, loss mask, OPUS
candidate -- is a SHA-256 over bytes produced here. Two rules make the hashes stable
across processes, machines and Python versions:

1. JSON is serialised canonically: keys sorted, no incidental whitespace, Unicode kept
   as Unicode (so Devanagari text hashes as itself rather than as \\u escapes).
2. Nothing hashed here depends on dict insertion order, set iteration order, `hash()`
   (which is salted per process), or floating-point repr drift.

`stable_uniform` deserves a note: OPUS needs a score that looks random across candidates
but is *reproducible*, because replay must reach the same decisions. Deriving it from a
hash of the candidate's own content gives that, whereas an RNG that advances with
training would not.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# Widget-facing id prefixes. The session's widgets render hashes with these tags
# (`tok_4d4543e296a4`, `sha256_9668bd19a4d9`, `clean_a1f0f9c8122a`,
# `lossmask_56b4985d`), and mirroring them keeps our artifacts readable against the
# schemas in resources/s6-widget-data.md.
SHORT_LEN = 12


def canonical_json(obj: Any) -> str:
    """Serialise `obj` so that equal objects always produce equal strings."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_bytes(obj: Any) -> bytes:
    return canonical_json(obj).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_bytes(obj))


def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def tagged(prefix: str, digest: str, length: int = SHORT_LEN) -> str:
    """`tagged("tok", "4d45...")` -> `tok_4d4543e296a4`, matching the session widgets."""
    return f"{prefix}_{digest[:length]}"


def short(digest: str, length: int = SHORT_LEN) -> str:
    return digest[:length]


def stable_uniform(*parts: Any) -> float:
    """A reproducible float in [0, 1) derived from the hash of `parts`.

    Used wherever the system needs a spread-out-but-deterministic quantity: OPUS proxy
    scores, synthetic quality signals. Never used for anything that should differ
    between two runs of the same configuration.
    """
    digest = hashlib.sha256(canonical_bytes(list(parts))).digest()
    # 53 bits keeps the result exactly representable as a float64.
    value = int.from_bytes(digest[:7], "big") >> 3
    return value / float(1 << 53)


def stable_choice(seq, *parts: Any):
    """Deterministically pick one element of `seq` from a hash of `parts`."""
    items = list(seq)
    if not items:
        raise ValueError("stable_choice on an empty sequence")
    return items[int(stable_uniform(*parts) * len(items)) % len(items)]
