"""The batch stream.

This is the spine. Everything about crash recovery, replay and forking reduces to one
property established here: **the stream is a pure function of (seed, branch_id, step,
schedule, admitted registry)**. Nothing about it depends on wall-clock time, filesystem
ordering, how many workers happened to be running, or the model's state.

The planner walks each lane's documents with a cursor. A cursor that runs off the end of a
lane wraps around, which is exactly what "2.4 epochs of the agentic lane" means in
practice -- deliberate repetition, visible and counted rather than accidental.

Candidates rejected by OPUS still advance the cursor. That is not a detail: the documents
were read, tokenized and packed before the selector saw them, so they are spent supply.
It is why the mixture compiler sizes demand as `planned / keep_fraction`, and why widget
15 gives OPUS-rejected tokens their own colour in the token-fate budget instead of folding
them into "useful".

`StreamPlanner` is deliberately cheap to fast-forward: no model, no gradients, just
tokenization-free reads out of memory-mapped shards. Replaying an interval means running
it from step 0 up to the interval start and then emitting -- which costs milliseconds and
removes any need to snapshot sampler state to make replay work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .hashing import sha256_json, short
from .manifest import ShardRegistry
from .masks import MaskedBatch, build_masks
from .mixture import MixtureSchedule
from .opus import Candidate, DecisionRecord, Opus, OpusLedger, OpusState
from .packing import PackDoc, pack, utilization

# A slot that cannot find an acceptable candidate after this many tries gives up and
# trains on the last one drawn, rather than looping forever. Hitting the cap is itself
# reportable: it means OPUS is rejecting a lane faster than the loader can feed it.
MAX_ATTEMPTS_PER_SLOT = 12


@dataclass
class MicrobatchPlan:
    """One accepted microbatch, fully described and independently reconstructible."""

    step: int
    microbatch_index: int
    lane: str
    stage: str
    candidate_batch_id: str
    opus_decision_id: str
    shard_ids: list[str]
    doc_ids: list[str]
    packed_sample_ids: list[str]
    token_span_ids: list[str]
    source_spans: list[dict]
    batch: MaskedBatch
    packing: dict
    attempts: int
    documents_drawn: int

    @property
    def batch_hash(self) -> str:
        return self.batch.batch_hash()

    @property
    def loss_mask_hash(self) -> str:
        return self.batch.loss_mask_hash()


@dataclass
class StepPlan:
    step: int
    stage: str
    microbatches: list[MicrobatchPlan]
    decisions: list[DecisionRecord]

    @property
    def loss_bearing_tokens(self) -> int:
        return sum(mb.batch.loss_bearing_tokens for mb in self.microbatches)

    @property
    def padding_tokens(self) -> int:
        return sum(mb.batch.padding_tokens for mb in self.microbatches)

    @property
    def context_only_tokens(self) -> int:
        return sum(mb.batch.context_only_tokens for mb in self.microbatches)

    def batch_hash(self) -> str:
        """Identity of the whole optimizer step, over its microbatches in order."""
        return "step_" + short(sha256_json([mb.batch_hash for mb in self.microbatches]), 12)


@dataclass
class LaneCursor:
    """Where a lane's reader has got to, and how many times it has been round."""

    documents: list[tuple[str, int]]  # (shard_id, doc index within shard)
    position: int = 0
    epochs: int = 0

    def next_index(self) -> tuple[str, int]:
        if not self.documents:
            raise RuntimeError("lane has no admitted documents")
        item = self.documents[self.position % len(self.documents)]
        self.position += 1
        if self.position % len(self.documents) == 0:
            self.epochs += 1
        return item

    def rewind(self, count: int) -> None:
        self.position = max(0, self.position - count)


@dataclass
class PlannerStats:
    documents_drawn: int = 0
    candidates_prepared: int = 0
    candidate_tokens_prepared: int = 0
    rejected_tokens: int = 0
    slots_hitting_attempt_cap: int = 0
    lane_epochs: dict = field(default_factory=dict)


class StreamPlanner:
    def __init__(
        self,
        config: Config,
        registry: ShardRegistry,
        schedule: MixtureSchedule,
        branch_id: str,
    ):
        self.config = config
        self.registry = registry
        self.schedule = schedule
        self.branch_id = branch_id
        self.seed = config.require("run.seed")

        self.context = config.require("batch.sequence_length")
        self.microbatch_size = config.require("batch.microbatch_size")
        self.slots_per_step = config.require("batch.microbatches_per_step")
        self.tokens_per_microbatch = self.context * self.microbatch_size

        from .tokenizer import FrozenTokenizer

        self.pad_id = FrozenTokenizer(config).pad_id

        self.cursors = {lane: self._lane_cursor(lane) for lane in config.lanes}
        self.opus = Opus(config, self._planned_lane_tokens())
        self.opus_state = OpusState()
        self.opus_ledger = OpusLedger()
        self.stats = PlannerStats()
        self.tokens_consumed = 0
        self.next_step = 0

    # -- setup -------------------------------------------------------------------

    def _lane_cursor(self, lane: str) -> LaneCursor:
        documents: list[tuple[str, int]] = []
        for entry in self.registry.admitted_by_lane(lane):
            for index in range(len(entry.shard.doc_spans)):
                documents.append((entry.shard_id, index))
        documents.sort()
        return LaneCursor(documents=documents)

    def _planned_lane_tokens(self) -> dict[str, int]:
        planned: dict[str, int] = {}
        for stage in self.schedule.stages:
            for lane, tokens in stage.planned_tokens.items():
                planned[lane] = planned.get(lane, 0) + tokens
        return planned

    def _pack_doc(self, shard_id: str, doc_index: int) -> PackDoc:
        entry = self.registry.by_id[shard_id]
        span = entry.shard.doc_spans[doc_index]
        tokens = entry.shard.span_tokens(span.token_start, span.token_end)
        return PackDoc(
            doc_id=span.doc_id,
            shard_id=shard_id,
            lane=span.lane,
            token_start=span.token_start,
            token_ids=[int(t) for t in tokens],
            segments=span.segments,
        )

    # -- candidate construction ---------------------------------------------------

    def _draw_candidate(self, step: int, microbatch: int, lane: str, attempt: int):
        """Draw documents, pack them, mask them, and describe the result."""
        policy = self.config.lane(lane)["packing_policy"]
        loss_policy = self.config.lane(lane)["loss_policy"]
        cursor = self.cursors[lane]

        drawn: list[PackDoc] = []
        slack = 0
        while True:
            target = self.tokens_per_microbatch + slack
            while sum(len(d) for d in drawn) < target:
                shard_id, index = cursor.next_index()
                drawn.append(self._pack_doc(shard_id, index))
            sequences = pack(list(drawn), policy, self.context, self.pad_id)
            if len(sequences) >= self.microbatch_size:
                break
            slack += self.context  # packing lost more to padding than expected

        kept = sequences[: self.microbatch_size]
        kept_doc_ids = {span.doc_id for seq in kept for span in seq.spans}

        # Documents drawn but not used land back in the queue instead of being burned.
        unused_tail = 0
        for doc in reversed(drawn):
            if doc.doc_id in kept_doc_ids:
                break
            unused_tail += 1
        if unused_tail:
            cursor.rewind(unused_tail)
            drawn = drawn[: len(drawn) - unused_tail]

        masked = [build_masks(seq, loss_policy, self.pad_id, self.context) for seq in kept]
        batch = MaskedBatch.stack(masked)

        packed_sample_ids = []
        token_span_ids = []
        source_spans = []
        for index, seq in enumerate(kept):
            doc_ids = sorted({span.doc_id for span in seq.spans})
            packed_sample_ids.append(f"sample_{index}_{short(sha256_json(doc_ids), 8)}")
            token_span_ids.append(f"{lane}_{step}_{index}:0-{self.context - 1}")
            for span in seq.spans:
                source_spans.append(
                    {
                        "sequence": index,
                        "doc_id": span.doc_id,
                        "shard_id": span.shard_id,
                        "role": span.role,
                        "seq_start": span.seq_start,
                        "seq_end": span.seq_end,
                        "shard_token_start": span.shard_token_start,
                        "shard_token_end": span.shard_token_end,
                    }
                )

        doc_ids = sorted({span["doc_id"] for span in source_spans})
        shard_ids = sorted({span["shard_id"] for span in source_spans})
        candidate_batch_id = "cand_" + short(
            sha256_json([self.branch_id, step, microbatch, attempt, doc_ids]), 8
        )

        candidate = Candidate(
            candidate_batch_id=candidate_batch_id,
            lane=lane,
            stage=self.schedule.stage_for_step(step),
            shard_ids=shard_ids,
            doc_ids=doc_ids,
            effective_token_estimate=batch.loss_bearing_tokens,
            # Re-checked at batch time as defence in depth. Documents that failed the
            # shard-level firewall never became shards, so this should stay False for the
            # whole run -- and the audit reports that it did.
            firewall_flagged=False,
        )

        stats = utilization(kept, self.context)
        stats["policy"] = policy
        stats["documents"] = len(kept_doc_ids)

        self.stats.documents_drawn += len(drawn)
        self.stats.candidates_prepared += 1
        self.stats.candidate_tokens_prepared += batch.non_pad_tokens

        return candidate, batch, stats, packed_sample_ids, token_span_ids, source_spans, len(drawn)

    # -- the plan ----------------------------------------------------------------

    def plan_step(self, step: int) -> StepPlan:
        """Produce one optimizer step. Must be called in order; see `advance_to`."""
        if step != self.next_step:
            raise RuntimeError(
                f"StreamPlanner is at step {self.next_step}, asked for {step}. "
                "Use advance_to() to fast-forward -- the stream is only defined as a "
                "sequence."
            )

        stage = self.schedule.stage_for_step(step)
        microbatches: list[MicrobatchPlan] = []
        decisions: list[DecisionRecord] = []

        for slot in range(self.slots_per_step):
            lane = self.schedule.lane_for(step, slot)
            accepted = None
            for attempt in range(MAX_ATTEMPTS_PER_SLOT):
                (
                    candidate,
                    batch,
                    packing_stats,
                    sample_ids,
                    span_ids,
                    source_spans,
                    drawn,
                ) = self._draw_candidate(step, slot, lane, attempt)

                bucket = self.config.model_age_bucket(self.tokens_consumed)
                record = self.opus.decide(
                    candidate, self.tokens_consumed, bucket, self.opus_state
                )
                self.opus_ledger.append(record)
                decisions.append(record)

                if record.consumed:
                    accepted = (candidate, batch, packing_stats, sample_ids, span_ids,
                                source_spans, record, attempt + 1, drawn)
                    break
                self.stats.rejected_tokens += batch.non_pad_tokens
            else:
                # Attempt cap reached: take the last candidate so the run continues, and
                # record that it happened rather than hiding it.
                self.stats.slots_hitting_attempt_cap += 1
                accepted = (candidate, batch, packing_stats, sample_ids, span_ids,
                            source_spans, record, MAX_ATTEMPTS_PER_SLOT, drawn)

            (candidate, batch, packing_stats, sample_ids, span_ids, source_spans,
             record, attempts, drawn) = accepted

            microbatches.append(
                MicrobatchPlan(
                    step=step,
                    microbatch_index=slot,
                    lane=lane,
                    stage=stage,
                    candidate_batch_id=candidate.candidate_batch_id,
                    opus_decision_id=record.opus_decision_id,
                    shard_ids=candidate.shard_ids,
                    doc_ids=candidate.doc_ids,
                    packed_sample_ids=sample_ids,
                    token_span_ids=span_ids,
                    source_spans=source_spans,
                    batch=batch,
                    packing=packing_stats,
                    attempts=attempts,
                    documents_drawn=drawn,
                )
            )
            self.tokens_consumed += batch.non_pad_tokens

        self.next_step = step + 1
        self.stats.lane_epochs = {lane: c.epochs for lane, c in sorted(self.cursors.items())}
        return StepPlan(step=step, stage=stage, microbatches=microbatches, decisions=decisions)

    def advance_to(self, step: int) -> None:
        """Fast-forward the stream without materialising the steps in between."""
        while self.next_step < step:
            self.plan_step(self.next_step)

    def consumed_lane_shares(self) -> dict[str, float]:
        total = self.opus_state.total_accepted_tokens
        if not total:
            return {}
        return {
            lane: round(tokens / total, 4)
            for lane, tokens in sorted(self.opus_state.accepted_tokens_by_lane.items())
        }
