"""Packing policies.

Widget 4 makes the cost of getting this wrong concrete: a document of 18 tokens in a
32-token context wastes 44% of the window under any padding policy, and 0% if the
remainder is filled with the next document. Widget 5 extends that across five policies and
shows the trade-off is not purely about utilisation -- concat-and-chop reaches 70% but
carries high boundary risk, while structure-preserving reaches 84% with low risk.

So all five policies are implemented, not just the one the demo uses, and each reports
utilisation, unused positions and boundary risk on our own corpus. The lane defaults sit
in `config/run_config.json`:

    web, indic          concat_and_chop        plain pretraining, EOS carries the boundary
    code                best_fit               keep whole files together where possible
    reasoning, agentic  structure_preserving   never split a sample or a role segment

"Boundary risk" is what a policy can do to a *sample*: whether a training example can be
cut in half so the model sees a conclusion without its premise, or an assistant turn
without the user turn it answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# What each policy can do to a document boundary.
BOUNDARY_RISK = {
    "pad_each_doc": "none",
    "concat_and_chop": "high",
    "greedy": "medium",
    "best_fit": "medium",
    "structure_preserving": "low",
}

# How far best-fit looks ahead for a document that fills the remaining space. Bounded so
# packing stays O(n * window) rather than quadratic in the candidate pool.
BEST_FIT_WINDOW = 24


@dataclass
class PackDoc:
    """One document offered to the packer, with its role structure intact."""

    doc_id: str
    shard_id: str
    lane: str
    token_start: int          # absolute offset within the shard
    token_ids: list[int]
    segments: list[dict]      # {role, start, end} relative to the document

    def __len__(self) -> int:
        return len(self.token_ids)


@dataclass
class PackedSpan:
    """A stretch of one sequence that came from one document segment."""

    doc_id: str
    shard_id: str
    role: str
    seq_start: int
    seq_end: int
    shard_token_start: int
    shard_token_end: int


@dataclass
class PackedSequence:
    token_ids: list[int] = field(default_factory=list)
    spans: list[PackedSpan] = field(default_factory=list)
    pad_count: int = 0

    @property
    def length(self) -> int:
        return len(self.token_ids)

    @property
    def used(self) -> int:
        return self.length - self.pad_count


def _emit_spans(
    doc: PackDoc, doc_offset: int, take: int, seq_cursor: int
) -> list[PackedSpan]:
    """Map the slice `doc[doc_offset : doc_offset + take]` onto sequence coordinates.

    Segments are clipped rather than dropped, so a document split across two sequences
    still reports which roles landed where -- the loss mask depends on it.
    """
    spans: list[PackedSpan] = []
    for segment in doc.segments:
        start = max(segment["start"], doc_offset)
        end = min(segment["end"], doc_offset + take)
        if start >= end:
            continue
        spans.append(
            PackedSpan(
                doc_id=doc.doc_id,
                shard_id=doc.shard_id,
                role=segment["role"],
                seq_start=seq_cursor + (start - doc_offset),
                seq_end=seq_cursor + (end - doc_offset),
                shard_token_start=doc.token_start + start,
                shard_token_end=doc.token_start + end,
            )
        )
    return spans


def _pad(sequence: PackedSequence, context: int, pad_id: int) -> PackedSequence:
    missing = context - sequence.length
    if missing > 0:
        sequence.token_ids.extend([pad_id] * missing)
        sequence.pad_count += missing
    return sequence


# ---------------------------------------------------------------------------
# The five policies
# ---------------------------------------------------------------------------


def pack_pad_each_doc(docs, context, pad_id):
    """One document per window. Boundary-safe, and mostly empty windows."""
    sequences = []
    for doc in docs:
        take = min(len(doc), context)
        seq = PackedSequence(token_ids=list(doc.token_ids[:take]))
        seq.spans = _emit_spans(doc, 0, take, 0)
        sequences.append(_pad(seq, context, pad_id))
    return sequences


def pack_concat_and_chop(docs, context, pad_id):
    """Concatenate everything and cut every `context` tokens.

    Efficient for plain pretraining, and the reason EOS matters: without it two unrelated
    documents on either side of a cut look like one continuous text.
    """
    sequences = []
    current = PackedSequence()
    for doc in docs:
        offset = 0
        while offset < len(doc):
            room = context - current.length
            take = min(room, len(doc) - offset)
            current.spans.extend(_emit_spans(doc, offset, take, current.length))
            current.token_ids.extend(doc.token_ids[offset : offset + take])
            offset += take
            if current.length == context:
                sequences.append(current)
                current = PackedSequence()
    if current.length:
        sequences.append(_pad(current, context, pad_id))
    return sequences


def pack_greedy(docs, context, pad_id):
    """Fill in arrival order with whole documents; close the window when the next won't fit."""
    sequences = []
    current = PackedSequence()
    for doc in docs:
        take = min(len(doc), context)
        if current.length + take > context:
            sequences.append(_pad(current, context, pad_id))
            current = PackedSequence()
        current.spans.extend(_emit_spans(doc, 0, take, current.length))
        current.token_ids.extend(doc.token_ids[:take])
    if current.length:
        sequences.append(_pad(current, context, pad_id))
    return sequences


def pack_best_fit(docs, context, pad_id):
    """Fill the tightest remaining gap first, looking a bounded distance ahead."""
    remaining = list(docs)
    sequences = []
    while remaining:
        current = PackedSequence()
        while True:
            room = context - current.length
            window = remaining[:BEST_FIT_WINDOW]
            # The largest document that still fits; ties broken by arrival order.
            choice = None
            for index, doc in enumerate(window):
                size = min(len(doc), context)
                if size <= room and (choice is None or size > choice[1]):
                    choice = (index, size)
            if choice is None:
                break
            index, size = choice
            doc = remaining.pop(index)
            current.spans.extend(_emit_spans(doc, 0, size, current.length))
            current.token_ids.extend(doc.token_ids[:size])
        if current.length == 0:
            # Nothing fits an empty window: the next document is longer than the context.
            doc = remaining.pop(0)
            current.spans.extend(_emit_spans(doc, 0, context, 0))
            current.token_ids.extend(doc.token_ids[:context])
        sequences.append(_pad(current, context, pad_id))
    return sequences


def pack_structure_preserving(docs, context, pad_id):
    """Never split a document, and never split a role segment.

    An over-long document is truncated at the last segment boundary that fits rather than
    mid-turn, so an agentic trajectory never loses the observation its answer depends on.
    """
    sequences = []
    current = PackedSequence()
    for doc in docs:
        take = len(doc)
        if take > context:
            take = 0
            for segment in doc.segments:
                if segment["end"] <= context:
                    take = max(take, segment["end"])
            if take == 0:
                continue  # not even one whole segment fits; drop rather than mangle
        if current.length + take > context:
            sequences.append(_pad(current, context, pad_id))
            current = PackedSequence()
        current.spans.extend(_emit_spans(doc, 0, take, current.length))
        current.token_ids.extend(doc.token_ids[:take])
    if current.length:
        sequences.append(_pad(current, context, pad_id))
    return sequences


POLICIES = {
    "pad_each_doc": pack_pad_each_doc,
    "concat_and_chop": pack_concat_and_chop,
    "greedy": pack_greedy,
    "best_fit": pack_best_fit,
    "structure_preserving": pack_structure_preserving,
}


def pack(docs: list[PackDoc], policy: str, context: int, pad_id: int) -> list[PackedSequence]:
    if policy not in POLICIES:
        raise ValueError(f"unknown packing policy {policy!r}; known: {sorted(POLICIES)}")
    return POLICIES[policy](docs, context, pad_id)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def utilization(sequences: list[PackedSequence], context: int) -> dict:
    positions = len(sequences) * context
    used = sum(s.used for s in sequences)
    return {
        "sequences": len(sequences),
        "positions": positions,
        "used_positions": used,
        "unused_positions": positions - used,
        "utilization": round(used / positions, 4) if positions else 0.0,
    }


def compare_policies(docs: list[PackDoc], context: int, pad_id: int) -> dict:
    """Widget 5's table, computed on our own documents rather than quoted from the widget."""
    rows = {}
    for name in sorted(POLICIES):
        sequences = pack(list(docs), name, context, pad_id)
        stats = utilization(sequences, context)
        stats["boundary_risk"] = BOUNDARY_RISK[name]
        rows[name] = stats
    return rows
