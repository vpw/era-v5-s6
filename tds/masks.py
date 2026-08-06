"""Loss masks, attention segmentation and position ids.

Session 1's contract, as widget 1 restates it: the model consumes fixed token windows and
learns next-token prediction, so sequence length, ordering, loss masks, attention policy
and EOS boundaries all have to survive packing intact.

Three arrays come out of every packed sequence, and they have to agree with each other:

  segment_ids    which packed document a position belongs to. Attention is allowed only
                 within a segment, so two unrelated documents sharing a window cannot
                 attend to each other.
  position_ids   position *within* the segment, reset at each document start. This is the
                 `packed_reset_on_eos` policy recorded in every ledger event.
  loss_mask      which positions are scored targets.

The loss mask is where the lanes differ, and widget 3 measures it: plain pretraining lanes
have every non-pad token loss-bearing, while an agentic trajectory has 12 non-pad tokens
and only 10 loss-bearing, because the user's request and the tool's observation are
context the model conditions on but is never scored against.

Two positions are never targets regardless of lane:

  * padding, which carries no information;
  * the first token of a segment, because predicting it would mean attending across a
    document boundary that `segment_ids` has already forbidden. Masking it is what keeps
    the loss mask consistent with the attention policy instead of quietly contradicting it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hashing import sha256_bytes, tagged
from .packing import PackedSequence

# Roles whose tokens the model is scored on when the lane is a supervised trajectory.
# `eos` is scored everywhere: knowing where a document ends is part of the task.
ASSISTANT_ROLES = frozenset({"assistant", "tool_call", "eos"})

LOSS_POLICIES = ("all_non_pad", "assistant_only")


@dataclass
class MaskedSequence:
    input_ids: np.ndarray      # (context,) int32
    loss_mask: np.ndarray      # (context,) uint8 -- 1 where the token is a scored target
    position_ids: np.ndarray   # (context,) int32 -- resets at each segment
    segment_ids: np.ndarray    # (context,) int32 -- -1 for padding
    roles: list[str]

    @property
    def context(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def non_pad_tokens(self) -> int:
        return int((self.segment_ids >= 0).sum())

    @property
    def loss_bearing_tokens(self) -> int:
        return int(self.loss_mask.sum())


def build_masks(
    sequence: PackedSequence, loss_policy: str, pad_id: int, context: int
) -> MaskedSequence:
    if loss_policy not in LOSS_POLICIES:
        raise ValueError(f"unknown loss policy {loss_policy!r}; known: {LOSS_POLICIES}")

    input_ids = np.asarray(sequence.token_ids, dtype=np.int32)
    if input_ids.shape[0] != context:
        raise ValueError(
            f"packed sequence has {input_ids.shape[0]} tokens, expected {context}"
        )

    segment_ids = np.full(context, -1, dtype=np.int32)
    position_ids = np.zeros(context, dtype=np.int32)
    loss_mask = np.zeros(context, dtype=np.uint8)
    roles = ["pad"] * context

    # Spans arrive in sequence order. A new segment starts whenever the document changes.
    current_doc: str | None = None
    segment_index = -1
    segment_start = 0

    for span in sequence.spans:
        if span.doc_id != current_doc:
            current_doc = span.doc_id
            segment_index += 1
            segment_start = span.seq_start
        for i in range(span.seq_start, span.seq_end):
            segment_ids[i] = segment_index
            position_ids[i] = i - segment_start
            roles[i] = span.role

    scored_roles = None if loss_policy == "all_non_pad" else ASSISTANT_ROLES
    for i in range(context):
        if segment_ids[i] < 0:
            continue                       # padding
        if position_ids[i] == 0:
            continue                       # no in-segment context to predict from
        if scored_roles is not None and roles[i] not in scored_roles:
            continue                       # context-only turn in a supervised trajectory
        loss_mask[i] = 1

    return MaskedSequence(
        input_ids=input_ids,
        loss_mask=loss_mask,
        position_ids=position_ids,
        segment_ids=segment_ids,
        roles=roles,
    )


def attention_mask(segment_ids: np.ndarray) -> np.ndarray:
    """Causal mask intersected with segment identity.

    True means "query may attend to key". Padding attends to nothing and is attended by
    nothing, so a padded window cannot leak into a real one.
    """
    context = segment_ids.shape[0]
    causal = np.tril(np.ones((context, context), dtype=bool))
    same_segment = segment_ids[:, None] == segment_ids[None, :]
    real = segment_ids >= 0
    return causal & same_segment & real[:, None] & real[None, :]


@dataclass
class MaskedBatch:
    """`microbatch_size` masked sequences, stacked."""

    input_ids: np.ndarray      # (b, context)
    loss_mask: np.ndarray      # (b, context)
    position_ids: np.ndarray   # (b, context)
    segment_ids: np.ndarray    # (b, context)

    @classmethod
    def stack(cls, sequences: list[MaskedSequence]) -> "MaskedBatch":
        return cls(
            input_ids=np.stack([s.input_ids for s in sequences]),
            loss_mask=np.stack([s.loss_mask for s in sequences]),
            position_ids=np.stack([s.position_ids for s in sequences]),
            segment_ids=np.stack([s.segment_ids for s in sequences]),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.input_ids.shape

    @property
    def non_pad_tokens(self) -> int:
        return int((self.segment_ids >= 0).sum())

    @property
    def loss_bearing_tokens(self) -> int:
        return int(self.loss_mask.sum())

    @property
    def padding_tokens(self) -> int:
        return int((self.segment_ids < 0).sum())

    @property
    def context_only_tokens(self) -> int:
        """Non-pad positions that carry no loss: masked turns and segment-initial tokens."""
        return self.non_pad_tokens - self.loss_bearing_tokens

    def loss_mask_hash(self) -> str:
        return tagged("lossmask", sha256_bytes(np.ascontiguousarray(self.loss_mask).tobytes()))

    def batch_hash(self) -> str:
        """Batch identity: what replay compares, independent of any model state.

        All four arrays participate. Two batches with the same tokens but different masks
        are different training events and must not hash alike.
        """
        parts = [
            np.ascontiguousarray(self.input_ids.astype(np.int32)).tobytes(),
            np.ascontiguousarray(self.loss_mask.astype(np.uint8)).tobytes(),
            np.ascontiguousarray(self.position_ids.astype(np.int32)).tobytes(),
            np.ascontiguousarray(self.segment_ids.astype(np.int32)).tobytes(),
        ]
        return tagged("batch", sha256_bytes(b"|".join(parts)))
