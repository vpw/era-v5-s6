"""Immutable tokenized shards.

A shard is three files that only ever get written once:

    <shard_id>.tokens.bin    uint32 little-endian token ids, one flat array
    <shard_id>.docs.jsonl    one record per document: its token span, its segment spans
                             with roles, its provenance and its safety state
    <shard_id>.manifest.json written by manifest.py

Immutability is not a convention here, it is enforced: `ShardWriter` refuses to write a
shard id that already exists on disk. That matters because replay reconstructs batches by
re-reading token spans out of these files months later -- if a shard could be rewritten in
place, a replayed batch would silently stop matching the batch that was actually trained
on, and every hash in the ledger would become a lie.

Documents are separated by EOS. The EOS token belongs to the span of the document it
terminates, so a span is self-delimiting and the packer can reset position ids on it.

Shards are grouped by (lane, metadata signature) rather than by lane alone. Admission is
decided per shard, so mixing documents with different safety metadata into one shard would
force the gate to judge the whole shard by its worst member. Segregating them means an
incomplete document quarantines itself instead of taking 200 good documents down with it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .corpus import Document
from .hashing import sha256_file, short, sha256_json, tagged
from .tokenizer import FrozenTokenizer

TOKEN_DTYPE = np.uint32


@dataclass
class DocSpan:
    """Where one document lives inside a shard's flat token array."""

    doc_id: str
    lane: str
    token_start: int
    token_end: int
    segments: list[dict]
    source: dict
    registry: dict

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start

    def to_record(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "lane": self.lane,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "token_count": self.token_count,
            "segments": self.segments,
            "source": self.source,
            "registry": self.registry,
        }


@dataclass
class Shard:
    shard_id: str
    lane: str
    tokens_path: Path
    docs_path: Path
    doc_spans: list[DocSpan]
    token_count: int
    content_hash: str
    metadata_signature: dict

    _tokens: np.ndarray | None = None

    def tokens(self) -> np.ndarray:
        """Memory-map the token array. Read-only: shards are immutable."""
        if self._tokens is None:
            self._tokens = np.memmap(
                self.tokens_path, dtype=TOKEN_DTYPE, mode="r", shape=(self.token_count,)
            )
        return self._tokens

    def span_tokens(self, token_start: int, token_end: int) -> np.ndarray:
        return np.asarray(self.tokens()[token_start:token_end])

    def verify_content_hash(self) -> bool:
        return tagged("sha256", sha256_file(self.tokens_path)) == self.content_hash


def metadata_signature(doc: Document) -> dict:
    """The safety/licence fields the admission gate reads, as a groupable key."""
    return {
        "license_tier": doc.registry.get("license_tier") or "unknown",
        "dedup_status": doc.registry.get("dedup_status"),
        "pii_screen_status": doc.registry.get("pii_screen_status"),
    }


class ShardWriter:
    def __init__(self, config: Config, tokenizer: FrozenTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.shard_dir = config.shard_dir
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.target_docs = config.require("corpus.shard_target_docs")

    def _shard_id(self, lane: str, doc_ids: list[str]) -> str:
        """Content-addressed id, so the same documents always produce the same shard."""
        digest = sha256_json({"lane": lane, "doc_ids": sorted(doc_ids)})
        return f"shard_{lane}_{short(digest, 8)}"

    def _group(self, docs: list[Document]) -> list[tuple[str, dict, list[Document]]]:
        buckets: dict[tuple, list[Document]] = {}
        for doc in docs:
            signature = metadata_signature(doc)
            key = (doc.lane, json.dumps(signature, sort_keys=True))
            buckets.setdefault(key, []).append(doc)

        groups: list[tuple[str, dict, list[Document]]] = []
        for (lane, signature_json), bucket in sorted(buckets.items()):
            bucket.sort(key=lambda d: d.doc_id)
            signature = json.loads(signature_json)
            for i in range(0, len(bucket), self.target_docs):
                groups.append((lane, signature, bucket[i : i + self.target_docs]))
        return groups

    def write_shard(self, lane: str, signature: dict, docs: list[Document]) -> Shard:
        shard_id = self._shard_id(lane, [d.doc_id for d in docs])
        tokens_path = self.shard_dir / f"{shard_id}.tokens.bin"
        docs_path = self.shard_dir / f"{shard_id}.docs.jsonl"

        if tokens_path.exists() or docs_path.exists():
            raise FileExistsError(
                f"shard {shard_id} already exists on disk. Shards are immutable: "
                "rebuilding a shard id with different content would invalidate every "
                "ledger entry and checkpoint that references it. Clear the work "
                "directory to rebuild from scratch."
            )

        token_stream: list[int] = []
        spans: list[DocSpan] = []
        eos = self.tokenizer.eos_id

        for doc in docs:
            doc_start = len(token_stream)
            segment_records: list[dict] = []
            for segment in self.tokenizer.encode_document(doc):
                seg_start = len(token_stream)
                token_stream.extend(segment.ids)
                if len(token_stream) > seg_start:
                    segment_records.append(
                        {
                            "role": segment.role,
                            "start": seg_start - doc_start,
                            "end": len(token_stream) - doc_start,
                        }
                    )
            if len(token_stream) == doc_start:
                continue  # a document that tokenized to nothing carries no signal

            # EOS closes the document and belongs to its span.
            eos_at = len(token_stream) - doc_start
            token_stream.append(eos)
            segment_records.append({"role": "eos", "start": eos_at, "end": eos_at + 1})

            spans.append(
                DocSpan(
                    doc_id=doc.doc_id,
                    lane=doc.lane,
                    token_start=doc_start,
                    token_end=len(token_stream),
                    segments=segment_records,
                    source=doc.source,
                    registry=doc.registry,
                )
            )

        array = np.asarray(token_stream, dtype=TOKEN_DTYPE)
        array.tofile(tokens_path)
        with open(docs_path, "w", encoding="utf-8") as fh:
            for span in spans:
                fh.write(json.dumps(span.to_record(), ensure_ascii=False, sort_keys=True) + "\n")

        return Shard(
            shard_id=shard_id,
            lane=lane,
            tokens_path=tokens_path,
            docs_path=docs_path,
            doc_spans=spans,
            token_count=int(array.size),
            content_hash=tagged("sha256", sha256_file(tokens_path)),
            metadata_signature=signature,
        )

    def write_all(self, docs: list[Document]) -> list[Shard]:
        shards = [
            self.write_shard(lane, signature, group)
            for lane, signature, group in self._group(docs)
        ]
        shards.sort(key=lambda s: s.shard_id)
        return shards


def load_shard(shard_dir: Path, shard_id: str) -> Shard:
    """Reopen a shard from disk alone -- the path replay and audit take."""
    tokens_path = Path(shard_dir) / f"{shard_id}.tokens.bin"
    docs_path = Path(shard_dir) / f"{shard_id}.docs.jsonl"
    spans: list[DocSpan] = []
    with open(docs_path, encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            spans.append(
                DocSpan(
                    doc_id=record["doc_id"],
                    lane=record["lane"],
                    token_start=record["token_start"],
                    token_end=record["token_end"],
                    segments=record["segments"],
                    source=record["source"],
                    registry=record["registry"],
                )
            )
    token_count = tokens_path.stat().st_size // np.dtype(TOKEN_DTYPE).itemsize
    lane = spans[0].lane if spans else shard_id.split("_")[1]
    return Shard(
        shard_id=shard_id,
        lane=lane,
        tokens_path=tokens_path,
        docs_path=docs_path,
        doc_spans=spans,
        token_count=token_count,
        content_hash=tagged("sha256", sha256_file(tokens_path)),
        metadata_signature={},
    )
