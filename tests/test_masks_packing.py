"""Packing policies, loss masks, attention segmentation and position ids.

The invariant tying these together: the three arrays must agree. If attention forbids
crossing a document boundary, the loss mask must not ask the model to predict across one,
and position ids must restart there. A system where they disagree still trains -- it just
trains on a task nobody described.
"""

from __future__ import annotations

import numpy as np
import pytest

from tds.masks import ASSISTANT_ROLES, MaskedBatch, attention_mask, build_masks
from tds.packing import (BOUNDARY_RISK, POLICIES, PackDoc, pack, utilization)

PAD = 3
CONTEXT = 32


def make_doc(doc_id: str, length: int, roles=None, start=0) -> PackDoc:
    tokens = list(range(100, 100 + length))
    if roles is None:
        segments = [{"role": "text", "start": 0, "end": length - 1},
                    {"role": "eos", "start": length - 1, "end": length}]
    else:
        segments = roles
    return PackDoc(doc_id=doc_id, shard_id="shard_test_0", lane="web",
                   token_start=start, token_ids=tokens, segments=segments)


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy", sorted(POLICIES))
def test_every_policy_produces_full_windows(policy):
    docs = [make_doc(f"d{i}", length) for i, length in enumerate([21, 8, 25, 12, 29, 16])]
    sequences = pack(docs, policy, CONTEXT, PAD)
    assert sequences, f"{policy} produced no sequences"
    for sequence in sequences:
        assert sequence.length == CONTEXT, f"{policy} produced a short window"


@pytest.mark.parametrize("policy", sorted(POLICIES))
def test_utilization_reconciles(policy):
    """Reported utilisation must equal used/positions, or the throughput claim is empty."""
    docs = [make_doc(f"d{i}", length) for i, length in enumerate([21, 38, 8, 25, 42, 12])]
    sequences = pack(docs, policy, CONTEXT, PAD)
    stats = utilization(sequences, CONTEXT)
    assert stats["used_positions"] + stats["unused_positions"] == stats["positions"]
    assert stats["utilization"] == pytest.approx(
        stats["used_positions"] / stats["positions"], abs=1e-4
    )


def test_padding_only_wastes_the_most():
    """Widget 4's finding, on our own packer: filling beats padding."""
    docs = [make_doc(f"d{i}", 18) for i in range(6)]
    padded = utilization(pack(docs, "pad_each_doc", CONTEXT, PAD), CONTEXT)
    packed = utilization(pack(docs, "concat_and_chop", CONTEXT, PAD), CONTEXT)
    assert packed["utilization"] > padded["utilization"]
    assert padded["unused_positions"] > packed["unused_positions"]


def test_structure_preserving_never_splits_a_document():
    docs = [make_doc(f"d{i}", length) for i, length in enumerate([20, 20, 20])]
    sequences = pack(docs, "structure_preserving", CONTEXT, PAD)
    for sequence in sequences:
        for doc_id in {span.doc_id for span in sequence.spans}:
            spans = [s for s in sequence.spans if s.doc_id == doc_id]
            covered = sum(s.seq_end - s.seq_start for s in spans)
            assert covered == 20, "a document was split across windows"


def test_concat_and_chop_does_split_documents():
    """The trade-off is real, and the boundary-risk labels should not be decorative."""
    docs = [make_doc(f"d{i}", 25) for i in range(4)]
    sequences = pack(docs, "concat_and_chop", CONTEXT, PAD)
    doc_windows = {}
    for index, sequence in enumerate(sequences):
        for span in sequence.spans:
            doc_windows.setdefault(span.doc_id, set()).add(index)
    assert any(len(windows) > 1 for windows in doc_windows.values())
    assert BOUNDARY_RISK["concat_and_chop"] == "high"
    assert BOUNDARY_RISK["structure_preserving"] == "low"


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def test_positions_reset_at_every_document_boundary():
    docs = [make_doc("a", 12), make_doc("b", 14)]
    sequence = pack(docs, "greedy", CONTEXT, PAD)[0]
    masked = build_masks(sequence, "all_non_pad", PAD, CONTEXT)

    starts = [i for i in range(CONTEXT) if masked.position_ids[i] == 0
              and masked.segment_ids[i] >= 0]
    assert len(starts) == 2, "each packed document should restart position ids"
    for index in range(1, CONTEXT):
        if masked.segment_ids[index] == masked.segment_ids[index - 1] >= 0:
            assert masked.position_ids[index] == masked.position_ids[index - 1] + 1


def test_attention_never_crosses_a_segment():
    docs = [make_doc("a", 12), make_doc("b", 14)]
    sequence = pack(docs, "greedy", CONTEXT, PAD)[0]
    masked = build_masks(sequence, "all_non_pad", PAD, CONTEXT)
    mask = attention_mask(masked.segment_ids)

    queries, keys = np.nonzero(mask)
    assert np.all(masked.segment_ids[queries] == masked.segment_ids[keys])
    assert np.all(queries >= keys), "attention must stay causal"
    assert not mask[masked.segment_ids < 0].any(), "padding must attend to nothing"


def test_segment_initial_tokens_are_never_targets():
    """Predicting a segment's first token would mean attending across a boundary."""
    docs = [make_doc("a", 12), make_doc("b", 14)]
    sequence = pack(docs, "greedy", CONTEXT, PAD)[0]
    masked = build_masks(sequence, "all_non_pad", PAD, CONTEXT)
    assert np.all(masked.loss_mask[masked.position_ids == 0] == 0)


def test_plain_lane_scores_every_non_pad_token_after_the_first():
    docs = [make_doc("a", 20)]
    sequence = pack(docs, "pad_each_doc", CONTEXT, PAD)[0]
    masked = build_masks(sequence, "all_non_pad", PAD, CONTEXT)
    assert masked.loss_bearing_tokens == masked.non_pad_tokens - 1


def test_agentic_lane_masks_context_turns():
    """Widget 3's key observation: loss-bearing < non-pad, only in the agentic lane."""
    segments = [
        {"role": "system", "start": 0, "end": 3},
        {"role": "user", "start": 3, "end": 8},
        {"role": "assistant", "start": 8, "end": 12},
        {"role": "tool_call", "start": 12, "end": 15},
        {"role": "tool_observation", "start": 15, "end": 20},
        {"role": "assistant", "start": 20, "end": 23},
        {"role": "eos", "start": 23, "end": 24},
    ]
    doc = make_doc("agent", 24, roles=segments)
    sequence = pack([doc], "structure_preserving", CONTEXT, PAD)[0]

    supervised = build_masks(sequence, "assistant_only", PAD, CONTEXT)
    plain = build_masks(sequence, "all_non_pad", PAD, CONTEXT)

    assert supervised.loss_bearing_tokens < supervised.non_pad_tokens
    assert supervised.loss_bearing_tokens < plain.loss_bearing_tokens
    scored_roles = {supervised.roles[i] for i in range(CONTEXT) if supervised.loss_mask[i]}
    assert scored_roles <= set(ASSISTANT_ROLES)
    assert "user" not in scored_roles and "tool_observation" not in scored_roles


def test_batch_hash_depends_on_the_mask_not_just_the_tokens():
    """Two batches with the same tokens but different masks are different events."""
    docs = [make_doc("a", 20)]
    sequence = pack(docs, "pad_each_doc", CONTEXT, PAD)[0]
    plain = MaskedBatch.stack([build_masks(sequence, "all_non_pad", PAD, CONTEXT)])
    supervised = MaskedBatch.stack([build_masks(sequence, "assistant_only", PAD, CONTEXT)])

    np.testing.assert_array_equal(plain.input_ids, supervised.input_ids)
    assert plain.batch_hash() != supervised.batch_hash()
    assert plain.loss_mask_hash() != supervised.loss_mask_hash()


def test_batch_hash_is_stable_across_rebuilds():
    docs = [make_doc("a", 20), make_doc("b", 9)]
    first = MaskedBatch.stack(
        [build_masks(s, "all_non_pad", PAD, CONTEXT) for s in pack(docs, "greedy", CONTEXT, PAD)]
    )
    second = MaskedBatch.stack(
        [build_masks(s, "all_non_pad", PAD, CONTEXT) for s in pack(docs, "greedy", CONTEXT, PAD)]
    )
    assert first.batch_hash() == second.batch_hash()
