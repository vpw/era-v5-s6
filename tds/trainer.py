"""The training loop, and the point where the data system meets a real gradient.

The loop itself is unremarkable. Two things about it are not.

**Rollback semantics.** A crash does not politely wait for a checkpoint boundary. When the
run dies at step 172 and the last checkpoint is at 150, recovery rolls the *model* back to
150 and re-consumes steps 150-171. That is not a repeated batch: the model state was rolled
back with the data, so each (model state, batch) pairing still occurs exactly once, which
is what "no skipped or repeated batches" actually means. Widget 14 shows precisely this --
rollback to ckpt-2 under ledger mode reproduces the same next batches.

To keep that auditable rather than merely asserted, every committed batch carries a
`recovery_epoch`. Events from before a rollback are superseded, not deleted -- the ledger
stays append-only -- and the verifier reconstructs the *effective* stream by taking the
highest recovery epoch for each step, then checks that for contiguity and duplicates.

**Loss attribution.** Per-token losses are mapped back through the packed spans to the
document and shard they came from, which is what lets the learning ledger say that
`indic-tier-a-042` is a strong late learner rather than just that the loss went down.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from .checkpoint import CheckpointMeta, CheckpointStore, checkpoint_id_for, registry_hash
from .config import Config
from .ledger import ConsumptionLedger
from .manifest import ShardRegistry
from .masks import attention_mask
from .mixture import MixtureSchedule
from .model import Adam, TinyTransformer, build_model
from .sampler import MicrobatchPlan, StreamPlanner, StepPlan
from .vocab import VocabProjection


@dataclass
class LossAttribution:
    """Per-shard and per-lane loss, accumulated as training proceeds."""

    by_shard: dict = field(default_factory=dict)
    by_lane: dict = field(default_factory=dict)

    def add(self, key_map: dict, shard_id: str, lane: str, stage: str,
            loss_sum: float, tokens: int, step: int) -> None:
        for scope, key in (("by_shard", shard_id), ("by_lane", lane)):
            table = getattr(self, scope)
            entry = table.setdefault(
                key,
                {"loss_sum": 0.0, "tokens": 0, "first_step": step, "last_step": step,
                 "lane": lane, "by_stage": {}},
            )
            entry["loss_sum"] += loss_sum
            entry["tokens"] += tokens
            entry["first_step"] = min(entry["first_step"], step)
            entry["last_step"] = max(entry["last_step"], step)
            stage_entry = entry["by_stage"].setdefault(stage, {"loss_sum": 0.0, "tokens": 0})
            stage_entry["loss_sum"] += loss_sum
            stage_entry["tokens"] += tokens


@dataclass
class Timings:
    data_seconds: float = 0.0
    compute_seconds: float = 0.0
    total_seconds: float = 0.0
    steps: int = 0


class Trainer:
    def __init__(
        self,
        config: Config,
        registry: ShardRegistry,
        schedule: MixtureSchedule,
        projection: VocabProjection,
        ledger: ConsumptionLedger,
        store: CheckpointStore,
        branch_id: str,
        recovery_epoch: int = 0,
        log=None,
    ):
        self.config = config
        self.registry = registry
        self.schedule = schedule
        self.projection = projection
        self.ledger = ledger
        self.store = store
        self.branch_id = branch_id
        self.recovery_epoch = recovery_epoch
        self.log = log or (lambda message: None)

        self.model, self.model_config = build_model(config, projection.size)
        self.optimizer = Adam(self.model.params, lr=config.require("train.learning_rate"))
        self.planner = StreamPlanner(config, registry, schedule, branch_id)

        self.checkpoint_interval = config.require("train.checkpoint_interval")
        self.warmup_steps = config.require("train.warmup_steps")
        self.grad_clip = config.require("train.grad_clip")
        self.position_policy = config.require("batch.position_policy")
        self.registry_hash = registry_hash(registry)

        self.attribution = LossAttribution()
        self.timings = Timings()
        self.step_losses: list[dict] = []
        self.last_checkpoint_id = "none"
        self.tokens_consumed = 0

    # -- helpers -----------------------------------------------------------------

    def _lr_scale(self, step: int) -> float:
        if self.warmup_steps and step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        return 1.0

    def _attribute(self, plan: MicrobatchPlan, per_token: np.ndarray, step: int) -> None:
        """Push per-position losses back onto the documents and shards they came from."""
        by_key: dict[tuple[str, str], list[float]] = {}
        for span in plan.source_spans:
            seq = span["sequence"]
            window = per_token[seq, span["seq_start"] : span["seq_end"]]
            if window.size == 0:
                continue
            scored = window[window > 0]
            if scored.size == 0:
                continue
            key = (span["shard_id"], span["doc_id"])
            bucket = by_key.setdefault(key, [0.0, 0])
            bucket[0] += float(scored.sum())
            bucket[1] += int(scored.size)

        for (shard_id, _doc_id), (loss_sum, tokens) in by_key.items():
            self.attribution.add({}, shard_id, plan.lane, plan.stage, loss_sum, tokens, step)

    def _train_microbatch(self, plan: MicrobatchPlan):
        batch = plan.batch
        input_ids = self.projection.project(batch.input_ids)
        masks = np.stack([attention_mask(batch.segment_ids[i]) for i in range(batch.shape[0])])
        loss, per_token, grads = self.model.forward_backward(
            input_ids, batch.position_ids, masks, batch.loss_mask
        )
        return loss, per_token, grads

    # -- the loop ----------------------------------------------------------------

    def run(self, start_step: int, end_step: int, crash_at_step: int | None = None) -> dict:
        """Train `[start_step, end_step)`. Returns a summary; may not return at all."""
        self.planner.advance_to(start_step)
        wall_start = time.perf_counter()

        for step in range(start_step, end_step):
            if crash_at_step is not None and step == crash_at_step:
                # The data for this step has been prepared but nothing is committed.
                # Kill the process outright: no atexit hooks, no flush of anything the
                # ledger did not already durably write. A resume that reads in-memory
                # state left over from the crashed run proves nothing.
                self.ledger.trainer_crashed(
                    global_step=step,
                    checkpoint_id=self.last_checkpoint_id,
                    failed_rank=self.config.require("recovery.crashed_rank"),
                    recovery_epoch=self.recovery_epoch,
                    note="deliberate crash injected by run_demo.py",
                )
                self.log(f"[CRASH] trainer_crashed at step {step}, last checkpoint "
                         f"{self.last_checkpoint_id}")
                os._exit(17)

            data_start = time.perf_counter()
            step_plan = self.planner.plan_step(step)
            data_time = time.perf_counter() - data_start
            self.timings.data_seconds += data_time

            compute_start = time.perf_counter()
            accumulated = {name: np.zeros_like(v) for name, v in self.model.params.items()}
            step_loss = 0.0
            scored_tokens = 0

            for plan in step_plan.microbatches:
                loss, per_token, grads = self._train_microbatch(plan)
                tokens = int(plan.batch.loss_bearing_tokens)
                step_loss += loss * tokens
                scored_tokens += tokens
                for name, grad in grads.items():
                    accumulated[name] += grad
                self._attribute(plan, per_token, step)

            scale = 1.0 / max(len(step_plan.microbatches), 1)
            for name in accumulated:
                accumulated[name] *= scale
            grad_norm = self.optimizer.step(
                self.model.params, accumulated, self._lr_scale(step), self.grad_clip
            )
            self.timings.compute_seconds += time.perf_counter() - compute_start

            mean_loss = step_loss / scored_tokens if scored_tokens else 0.0
            self.tokens_consumed += scored_tokens

            # One batch_committed per optimizer step, carrying its microbatches -- the
            # shape widget 8 records, with microbatch_count and one sample id per
            # microbatch.
            self.ledger.batch_committed(
                global_step=step,
                checkpoint_id=self.last_checkpoint_id,
                rank=len(step_plan.microbatches),
                microbatch_count=len(step_plan.microbatches),
                packed_sample_ids=[
                    sid for mb in step_plan.microbatches for sid in mb.packed_sample_ids
                ],
                shard_ids=sorted({s for mb in step_plan.microbatches for s in mb.shard_ids}),
                token_span_ids=[
                    tid for mb in step_plan.microbatches for tid in mb.token_span_ids
                ],
                loss_mask_hash=step_plan.microbatches[0].loss_mask_hash,
                position_policy=self.position_policy,
                mixture_lane=step_plan.microbatches[0].lane,
                curriculum_stage=step_plan.stage,
                opus_decision_id=step_plan.microbatches[0].opus_decision_id,
                batch_hash=step_plan.batch_hash(),
                packing_utilization=round(
                    sum(mb.packing["utilization"] for mb in step_plan.microbatches)
                    / len(step_plan.microbatches),
                    4,
                ),
                loss_bearing_tokens=scored_tokens,
                microbatch_lanes=[mb.lane for mb in step_plan.microbatches],
                microbatch_hashes=[mb.batch_hash for mb in step_plan.microbatches],
                mean_loss=round(mean_loss, 6),
                grad_norm=round(grad_norm, 6),
                recovery_epoch=self.recovery_epoch,
            )

            self.step_losses.append(
                {"step": step, "stage": step_plan.stage, "loss": round(mean_loss, 6),
                 "tokens": scored_tokens, "grad_norm": round(grad_norm, 6)}
            )
            self.timings.steps += 1

            if (step + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(step + 1)

        self.timings.total_seconds += time.perf_counter() - wall_start
        return self.summary()

    def save_checkpoint(self, next_step: int, parent: dict | None = None) -> str:
        checkpoint_id = checkpoint_id_for(next_step)
        meta = CheckpointMeta(
            checkpoint_id=checkpoint_id,
            global_step=next_step - 1,
            next_step=next_step,
            ledger_offset=self.ledger.next_offset,
            branch_id=self.branch_id,
            model_hash=self.model.state_hash(),
            config_sha256=self.config.config_hash,
            registry_hash=self.registry_hash,
            tokens_consumed=self.tokens_consumed,
            parent_checkpoint_id=(parent or {}).get("checkpoint_id"),
            parent_branch_id=(parent or {}).get("branch_id"),
        )
        self.store.save(checkpoint_id, self.model, self.optimizer, meta)
        self.ledger.checkpoint_bound(
            global_step=next_step - 1,
            checkpoint_id=checkpoint_id,
            dataloader_state=f"ledger_offset_{meta.ledger_offset}",
            next_step=next_step,
            recovery_epoch=self.recovery_epoch,
        )
        self.last_checkpoint_id = checkpoint_id
        self.log(f"[PASS] checkpoint_saved {checkpoint_id} at step {next_step - 1} "
                 f"bound to ledger_offset_{meta.ledger_offset}")
        return checkpoint_id

    def restore(self, checkpoint_id: str) -> dict:
        record = self.store.load(checkpoint_id, self.model, self.optimizer)
        self.last_checkpoint_id = checkpoint_id
        self.tokens_consumed = record["tokens_consumed"]
        return record

    def summary(self) -> dict:
        return {
            "branch_id": self.branch_id,
            "steps_trained": self.timings.steps,
            "tokens_consumed": self.tokens_consumed,
            "final_model_hash": self.model.state_hash(),
            "parameters": self.model.parameter_count,
            "timings": {
                "data_seconds": round(self.timings.data_seconds, 4),
                "compute_seconds": round(self.timings.compute_seconds, 4),
                "total_seconds": round(self.timings.total_seconds, 4),
            },
            "opus": self.planner.opus_ledger.summary(),
            "planner": {
                "documents_drawn": self.planner.stats.documents_drawn,
                "candidates_prepared": self.planner.stats.candidates_prepared,
                "positions_prepared": self.planner.stats.positions_prepared,
                "positions_rejected": self.planner.stats.positions_rejected,
                "nonpad_prepared": self.planner.stats.nonpad_prepared,
                "slots_hitting_attempt_cap": self.planner.stats.slots_hitting_attempt_cap,
                "lane_epochs": self.planner.stats.lane_epochs,
            },
            "consumed_lane_shares": self.planner.consumed_lane_shares(),
        }
