"""OPUS: which prepared batches actually get trained on, and why.

Four outcomes, following widget 10: `accepted`, `rejected`, `deferred`, `protected`. Every
one produces a decision record with the same shape, so the audit trail is uniform and a
rejection is as inspectable as an acceptance. Rejection reasons come from the widget's
taxonomy, and they are kept distinct on purpose -- "the lane quota is full", "the proxy
thinks this is weak" and "this touches the eval set" mean completely different things when
V6 data planning reads this log back.

Two rules matter more than the rest.

**The proxy score is derived from the candidate's own content hash**, crossed with lane,
stage and model age. It is never drawn from an RNG that advances with training. If it
were, replaying an interval would reach different decisions, the replayed batch stream
would diverge from the original, and every hash in the ledger would stop matching. The
selector has to be as reproducible as the sampler.

**A protected floor never overrides the eval firewall.** Widget 10 shows this directly:
an Indic candidate scoring 0.518 under a 0.90 threshold gets rescued into the protected
ledger, while a different Indic candidate that failed on `eval_firewall_overlap` stays
rejected. Floors and the firewall are independent gates and the firewall always wins --
starving a lane is a planning problem, and training on the test set is not.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .hashing import short, sha256_json, stable_uniform

ACCEPTED = "accepted"
REJECTED = "rejected"
DEFERRED = "deferred"
PROTECTED = "protected"

# Widget 10's five rejection reasons, plus the two non-rejection outcomes that reuse the
# same field.
STAGE_MISMATCH = "stage_mismatch"
DUPLICATE_UPDATE = "duplicate_update_direction"
EVAL_FIREWALL = "eval_firewall_overlap"
BELOW_THRESHOLD = "below_proxy_threshold"
LANE_QUOTA_FULL = "lane_quota_full"
DEFERRED_FOR_ANNEAL = "deferred_for_anneal"
PROTECTED_FLOOR_OVERRIDE = "protected_floor_override"

# Reasons a protected floor may overturn. `eval_firewall_overlap` is deliberately absent.
RESCUABLE = frozenset({STAGE_MISMATCH, DUPLICATE_UPDATE, BELOW_THRESHOLD, LANE_QUOTA_FULL})

# How far back "the same update direction" is remembered.
DUPLICATE_WINDOW = 32

# Stage affinity: how much a lane's proxy score is nudged in each curriculum stage. Web
# earns its keep early and loses value late; reasoning is the reverse. This is the
# session's "a high-perplexity token at 400M tokens can be healthy novelty, the same token
# at 5.6B may indicate missing data" idea applied to whole batches.
STAGE_AFFINITY = {
    "foundation":  {"web": 0.10, "code": 0.02, "indic": 0.00, "reasoning": -0.10, "agentic": -0.08},
    "skill_build": {"web": -0.04, "code": 0.06, "indic": 0.02, "reasoning": 0.06, "agentic": 0.02},
    "anneal":      {"web": -0.12, "code": 0.00, "indic": 0.08, "reasoning": 0.10, "agentic": 0.06},
}


@dataclass
class Candidate:
    """A prepared batch offered to OPUS, before any gradient has been computed."""

    candidate_batch_id: str
    lane: str
    stage: str
    shard_ids: list[str]
    doc_ids: list[str]
    effective_token_estimate: int
    firewall_flagged: bool = False

    @property
    def update_signature(self) -> str:
        """Identity of the *update direction*: which documents this batch would train on.

        Two batches over the same documents push the weights the same way, so seeing the
        signature again soon is a duplicate update rather than new information.
        """
        return short(sha256_json(sorted(self.doc_ids)), 12)


@dataclass
class DecisionRecord:
    opus_decision_id: str
    candidate_batch_id: str
    model_age: str
    proxy_version: str
    lane: str
    stage: str
    score: float
    decision: str
    rejection_reason: str | None
    shard_ids: list[str]
    effective_token_estimate: int

    @property
    def consumed(self) -> bool:
        """Accepted and protected batches are trained on; rejected and deferred are not."""
        return self.decision in (ACCEPTED, PROTECTED)

    def to_record(self) -> dict:
        return {
            "opus_decision_id": self.opus_decision_id,
            "candidate_batch_id": self.candidate_batch_id,
            "model_age": self.model_age,
            "proxy_version": self.proxy_version,
            "lane": self.lane,
            "stage": self.stage,
            "score": self.score,
            "decision": self.decision,
            "rejection_reason": self.rejection_reason,
            "shard_ids": list(self.shard_ids),
            "effective_token_estimate": self.effective_token_estimate,
        }


@dataclass
class OpusState:
    """Everything the selector remembers. A pure function of the decision prefix.

    Because it is only a function of the prefix, replaying an interval means replaying the
    selector from the start of the run -- which costs nothing, since no model is involved.
    """

    accepted_tokens_by_lane: dict[str, int] = field(default_factory=dict)
    total_accepted_tokens: int = 0
    decisions: int = 0
    recent_signatures: deque = field(default_factory=lambda: deque(maxlen=DUPLICATE_WINDOW))
    counts: dict[str, int] = field(default_factory=dict)
    reasons: dict[str, int] = field(default_factory=dict)

    def lane_share(self, lane: str) -> float:
        if not self.total_accepted_tokens:
            return 0.0
        return self.accepted_tokens_by_lane.get(lane, 0) / self.total_accepted_tokens


class Opus:
    def __init__(self, config: Config, planned_lane_tokens: dict[str, int]):
        self.proxy_version = config.require("opus.proxy_version")
        self.accept_threshold = config.require("opus.accept_threshold")
        self.defer_threshold = config.require("opus.defer_threshold")
        self.quota_slack = config.require("opus.lane_quota_slack")
        self.floors = config.require("mixture.protected_floors")
        self.anneal_reserve_lanes = set(config.require("mixture.anneal_reserve_lanes"))
        self.stage_weights = config.require("mixture.stage_weights")
        self.planned_lane_tokens = planned_lane_tokens
        self.seed = config.require("run.seed")

    # -- the proxy ---------------------------------------------------------------

    def score(self, candidate: Candidate, model_age_bucket: str) -> float:
        base = stable_uniform(
            self.proxy_version,
            self.seed,
            candidate.candidate_batch_id,
            candidate.lane,
            model_age_bucket,
        )
        affinity = STAGE_AFFINITY.get(candidate.stage, {}).get(candidate.lane, 0.0)
        return round(min(max(base + affinity, 0.0), 1.0), 3)

    # -- the gate ----------------------------------------------------------------

    def _reason_for(self, candidate: Candidate, score: float, state: OpusState) -> str | None:
        """First failing gate, in priority order. The firewall is checked first."""
        if candidate.firewall_flagged:
            return EVAL_FIREWALL

        stage_weight = self.stage_weights.get(candidate.stage, {}).get(candidate.lane, 0.0)
        if stage_weight <= 0.0:
            return STAGE_MISMATCH

        if candidate.update_signature in state.recent_signatures:
            return DUPLICATE_UPDATE

        planned = self.planned_lane_tokens.get(candidate.lane, 0)
        consumed = state.accepted_tokens_by_lane.get(candidate.lane, 0)
        if planned and consumed > planned * self.quota_slack:
            return LANE_QUOTA_FULL

        if score < self.accept_threshold:
            return BELOW_THRESHOLD

        return None

    def decide(
        self, candidate: Candidate, tokens_seen: int, model_age_bucket: str, state: OpusState
    ) -> DecisionRecord:
        score = self.score(candidate, model_age_bucket)
        reason = self._reason_for(candidate, score, state)

        if reason is None:
            decision, rejection_reason = ACCEPTED, None
        elif reason == EVAL_FIREWALL:
            # Not rescuable under any circumstances. This branch exists before the
            # protected-floor branch so that ordering cannot be changed by accident.
            decision, rejection_reason = REJECTED, EVAL_FIREWALL
        elif self._below_floor(candidate.lane, state) and reason in RESCUABLE:
            decision, rejection_reason = PROTECTED, PROTECTED_FLOOR_OVERRIDE
        elif (
            reason == BELOW_THRESHOLD
            and score >= self.defer_threshold
            and candidate.lane in self.anneal_reserve_lanes
            and candidate.stage != "anneal"
        ):
            # Strong-but-not-accepted material in a reserved lane is held for the anneal
            # rather than spent early. Widget 7's "anneal reserve" as a live decision.
            decision, rejection_reason = DEFERRED, DEFERRED_FOR_ANNEAL
        else:
            decision, rejection_reason = REJECTED, reason

        record = DecisionRecord(
            opus_decision_id=f"opus_{state.decisions:05d}",
            candidate_batch_id=candidate.candidate_batch_id,
            model_age=f"{model_age_bucket} {tokens_seen} tokens",
            proxy_version=self.proxy_version,
            lane=candidate.lane,
            stage=candidate.stage,
            score=score,
            decision=decision,
            rejection_reason=rejection_reason,
            shard_ids=sorted(set(candidate.shard_ids)),
            effective_token_estimate=candidate.effective_token_estimate,
        )
        self._apply(record, candidate, state)
        return record

    def _below_floor(self, lane: str, state: OpusState) -> bool:
        floor = self.floors.get(lane)
        if floor is None:
            return False
        # Before anything has been accepted every lane is trivially "below floor"; that
        # would rescue the first candidate of every protected lane regardless of merit.
        if state.total_accepted_tokens == 0:
            return False
        return state.lane_share(lane) < floor

    def _apply(self, record: DecisionRecord, candidate: Candidate, state: OpusState) -> None:
        state.decisions += 1
        state.counts[record.decision] = state.counts.get(record.decision, 0) + 1
        key = record.rejection_reason or "none"
        state.reasons[key] = state.reasons.get(key, 0) + 1
        if record.consumed:
            state.accepted_tokens_by_lane[candidate.lane] = (
                state.accepted_tokens_by_lane.get(candidate.lane, 0)
                + candidate.effective_token_estimate
            )
            state.total_accepted_tokens += candidate.effective_token_estimate
            state.recent_signatures.append(candidate.update_signature)


class OpusLedger:
    """The decision board: every record, in order, written out as an artifact."""

    def __init__(self):
        self.records: list[DecisionRecord] = []

    def append(self, record: DecisionRecord) -> None:
        self.records.append(record)

    def by_decision(self, decision: str) -> list[DecisionRecord]:
        return [r for r in self.records if r.decision == decision]

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        reasons: dict[str, int] = {}
        for record in self.records:
            counts[record.decision] = counts.get(record.decision, 0) + 1
            key = record.rejection_reason or "none"
            reasons[key] = reasons.get(key, 0) + 1
        total = len(self.records)
        consumed = sum(1 for r in self.records if r.consumed)
        return {
            "decisions": total,
            "by_decision": counts,
            "by_rejection_reason": reasons,
            "consumed": consumed,
            "rejection_rate": round(1.0 - consumed / total, 4) if total else 0.0,
            "tokens_rejected": sum(
                r.effective_token_estimate for r in self.records if not r.consumed
            ),
        }

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for record in self.records:
                fh.write(json.dumps(record.to_record(), ensure_ascii=False, sort_keys=True) + "\n")
        return path
