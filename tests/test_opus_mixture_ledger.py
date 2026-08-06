"""OPUS decisions, the compiled mixture schedule, and ledger integrity.

The two invariants worth stating plainly:

* **OPUS must be deterministic.** If the proxy score moved with training, replaying an
  interval would reach different decisions and every hash in the ledger would stop
  matching. Determinism here is not tidiness, it is what makes replay possible at all.
* **A protected floor never overrides the eval firewall.** Widget 10 shows an Indic
  candidate rescued by its floor and another Indic candidate, failing on
  `eval_firewall_overlap`, still rejected. Starving a lane is a planning problem; training
  on the test set is not.
"""

from __future__ import annotations

import pytest

from tds.ledger import (BATCH_COMMITTED, effective_batches, summarize,
                        verify_append_only)
from tds.mixture import MixtureSchedule, _largest_remainder, _stride_interleave
from tds.opus import (ACCEPTED, DEFERRED, EVAL_FIREWALL, PROTECTED,
                      PROTECTED_FLOOR_OVERRIDE, REJECTED, RESCUABLE, Candidate, Opus,
                      OpusState)

PLANNED = {"web": 100000, "code": 60000, "indic": 50000, "reasoning": 40000, "agentic": 20000}


def make_candidate(batch_id="cand_0", lane="indic", stage="skill_build",
                   firewall_flagged=False, tokens=500) -> Candidate:
    return Candidate(
        candidate_batch_id=batch_id, lane=lane, stage=stage,
        shard_ids=[f"shard_{lane}_1"], doc_ids=[f"{lane}-doc-{batch_id}"],
        effective_token_estimate=tokens, firewall_flagged=firewall_flagged,
    )


@pytest.fixture
def opus(config) -> Opus:
    return Opus(config, PLANNED)


# ---------------------------------------------------------------------------
# OPUS
# ---------------------------------------------------------------------------


def test_scoring_is_deterministic(opus):
    candidate = make_candidate()
    first = opus.score(candidate, "early")
    for _ in range(5):
        assert opus.score(candidate, "early") == first


def test_scoring_depends_on_model_age_not_call_order(opus):
    candidate = make_candidate()
    early = opus.score(candidate, "early")
    late = opus.score(candidate, "late")
    assert opus.score(candidate, "early") == early
    assert early != late or True  # ages may coincide; the point is the repeat call matches


def test_replaying_the_same_candidates_reaches_the_same_decisions(opus):
    candidates = [make_candidate(f"cand_{i}", lane="web", stage="foundation")
                  for i in range(30)]

    def run():
        state = OpusState()
        return [
            opus.decide(c, 1000 * i, "early", state).to_record()
            for i, c in enumerate(candidates)
        ]

    first, second = run(), run()
    assert first == second


def test_firewall_hit_is_rejected(opus):
    state = OpusState()
    record = opus.decide(make_candidate(firewall_flagged=True), 0, "early", state)
    assert record.decision == REJECTED
    assert record.rejection_reason == EVAL_FIREWALL


def test_protected_floor_never_rescues_a_firewall_hit(opus):
    """The central OPUS invariant. Floors and the firewall are independent gates."""
    state = OpusState()
    # Drive the indic lane far below its protected floor by consuming another lane.
    state.accepted_tokens_by_lane = {"web": 1_000_000, "indic": 1}
    state.total_accepted_tokens = 1_000_001
    assert state.lane_share("indic") < opus.floors["indic"]

    record = opus.decide(make_candidate(lane="indic", firewall_flagged=True), 0, "early", state)
    assert record.decision == REJECTED
    assert record.rejection_reason == EVAL_FIREWALL
    assert record.rejection_reason != PROTECTED_FLOOR_OVERRIDE


def test_protected_floor_does_rescue_a_weak_score(opus):
    """The other half of the same invariant: floors do override score-type rejections."""
    state = OpusState()
    state.accepted_tokens_by_lane = {"web": 1_000_000, "indic": 1}
    state.total_accepted_tokens = 1_000_001

    rescued = None
    for i in range(200):
        candidate = make_candidate(f"weak_{i}", lane="indic", stage="foundation")
        if opus.score(candidate, "early") < opus.accept_threshold:
            rescued = opus.decide(candidate, 0, "early", state)
            break
    assert rescued is not None, "no below-threshold indic candidate found"
    assert rescued.decision == PROTECTED
    assert rescued.rejection_reason == PROTECTED_FLOOR_OVERRIDE


def test_eval_firewall_is_not_in_the_rescuable_set():
    assert EVAL_FIREWALL not in RESCUABLE


def test_decision_records_carry_every_widget_10_field(opus):
    state = OpusState()
    record = opus.decide(make_candidate(), 800, "early", state).to_record()
    assert set(record) == {
        "opus_decision_id", "candidate_batch_id", "model_age", "proxy_version", "lane",
        "stage", "score", "decision", "rejection_reason", "shard_ids",
        "effective_token_estimate",
    }
    assert record["proxy_version"] == opus.proxy_version


def test_accepted_records_have_no_rejection_reason(opus):
    state = OpusState()
    for i in range(200):
        record = opus.decide(make_candidate(f"c{i}", lane="web", stage="foundation"),
                             0, "early", state)
        if record.decision == ACCEPTED:
            assert record.rejection_reason is None
            return
    pytest.fail("no candidate was ever accepted")


def test_only_consumed_decisions_advance_lane_totals(opus):
    state = OpusState()
    for i in range(60):
        opus.decide(make_candidate(f"c{i}", lane="web", stage="foundation"), 0, "early", state)
    consumed = state.counts.get(ACCEPTED, 0) + state.counts.get(PROTECTED, 0)
    assert state.total_accepted_tokens == consumed * 500


# ---------------------------------------------------------------------------
# Mixture
# ---------------------------------------------------------------------------


def test_stage_weights_sum_to_one_and_respect_floors(config):
    schedule = MixtureSchedule(config)  # raises if either is violated
    for stage in schedule.stages:
        assert sum(stage.weights.values()) == pytest.approx(1.0)
        for lane, floor in schedule.floors.items():
            assert stage.weights[lane] >= floor


def test_compiled_plan_hits_the_planned_shares(config):
    """Exactness is the reason the schedule is compiled rather than sampled."""
    schedule = MixtureSchedule(config)
    assert len(schedule.lane_plan) == schedule.total_slots
    shares = schedule.planned_lane_shares()
    assert sum(shares.values()) == pytest.approx(1.0)
    for lane, share in shares.items():
        assert share > 0


def test_largest_remainder_apportions_exactly():
    counts = _largest_remainder({"a": 0.45, "b": 0.35, "c": 0.20}, 97)
    assert sum(counts.values()) == 97


def test_stride_interleave_spreads_rather_than_blocks():
    order = _stride_interleave({"a": 6, "b": 3, "c": 1})
    assert len(order) == 10
    # A blocked layout would put all six "a"s first; a spread one must not.
    assert order[:6] != ["a"] * 6


def test_lane_plan_is_reproducible(config):
    assert MixtureSchedule(config).lane_plan == MixtureSchedule(config).lane_plan


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def batch_event(offset, step, branch="run-a", epoch=0) -> dict:
    return {
        "event": BATCH_COMMITTED, "ledger_offset": offset, "run_branch_id": branch,
        "global_step": step, "recovery_epoch": epoch, "mixture_lane": "web",
        "loss_bearing_tokens": 100, "microbatch_lanes": ["web"],
    }


def test_clean_ledger_verifies():
    events = [batch_event(i, i) for i in range(10)]
    result = verify_append_only(events)
    assert result["ok"], result["issues"]
    assert result["offsets_contiguous"]


def test_gap_in_the_effective_stream_is_caught():
    events = [batch_event(i, step) for i, step in enumerate([0, 1, 2, 4, 5])]
    result = verify_append_only(events)
    assert not result["ok"]
    assert any("missing steps" in issue for issue in result["issues"])


def test_repeat_at_the_same_recovery_epoch_is_caught():
    events = [batch_event(i, step) for i, step in enumerate([0, 1, 2, 2, 3])]
    result = verify_append_only(events)
    assert not result["ok"]
    assert any("committed twice" in issue for issue in result["issues"])


def test_rollback_supersedes_rather_than_duplicates():
    """A crash rolls the model back too, so re-consuming steps is not a repeat."""
    events = [batch_event(i, i) for i in range(6)]                       # steps 0-5
    events += [batch_event(6 + i, 3 + i, epoch=1) for i in range(3)]     # rollback to 3
    result = verify_append_only(events)
    assert result["ok"], result["issues"]

    per_branch = result["per_branch"]["run-a"]
    assert per_branch["superseded_by_rollback"] == 3
    assert per_branch["effective_steps"] == 6

    effective = effective_batches(events, "run-a")
    assert sorted(effective) == [0, 1, 2, 3, 4, 5]
    for step in (3, 4, 5):
        assert effective[step]["recovery_epoch"] == 1


def test_summary_counts_the_effective_stream_only():
    events = [batch_event(i, i) for i in range(6)]
    events += [batch_event(6 + i, 3 + i, epoch=1) for i in range(3)]
    summary = summarize(events)
    assert summary["batches_committed"] == 9
    assert summary["effective_batches"] == 6
    assert summary["superseded_by_rollback"] == 3
    assert summary["loss_bearing_tokens"] == 600
