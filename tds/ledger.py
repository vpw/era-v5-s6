"""The training consumption ledger.

Widget 8 states the principle this file exists to enforce: *the ledger stores facts after
consumption, not just intentions before training*. A schedule says what should happen. The
ledger says what did.

It is append-only in the strict sense -- events are opened in append mode, flushed on
write, and never edited or reordered. `ledger_offset` is the index of the event in the log,
so it counts every event and not just batches, matching widget 8's own numbering where a
`checkpoint_bound` at offset 7 follows batches at offsets 0-3.

Event schemas come from widgets 8 and 14 verbatim, with three fields added that the
assignment's replay proof needs: `batch_hash`, `packing_utilization` and
`loss_bearing_tokens`. `run_branch_id` changes only on an intentional fork.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

# Widget 8's three consumption events plus widget 14's three run-level events.
BATCH_COMMITTED = "batch_committed"
CHECKPOINT_BOUND = "checkpoint_bound"
WORKER_CRASH_RECOVERED = "worker_crash_recovered"
RUN_STARTED = "run_started"
TRAINER_CRASHED = "trainer_crashed"
CHECKPOINT_RESTORED = "checkpoint_restored"

# Widget 14's recovery vocabulary. `ledger` replays the recorded stream, `fork` starts a
# new branch from the same checkpoint, `random` is what happens with no ledger at all.
MODE_LEDGER = "ledger"
MODE_FORK = "fork"
MODE_RANDOM = "random"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class ConsumptionLedger:
    """Append-only event log. One file, one run, many branches."""

    def __init__(self, path: Path, branch_id: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.branch_id = branch_id
        self.events: list[dict] = []
        if self.path.exists():
            self.events = read_events(self.path)

    @property
    def next_offset(self) -> int:
        return len(self.events)

    def append(self, event: str, **fields) -> dict:
        record = {
            "event": event,
            "ledger_offset": self.next_offset,
            "run_branch_id": self.branch_id,
            "created_at": utc_now(),
            **fields,
        }
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()
        self.events.append(record)
        return record

    # -- the event types ---------------------------------------------------------

    def run_started(self, *, global_step: int, config_sha256: str, mode: str, **extra) -> dict:
        return self.append(
            RUN_STARTED,
            global_step=global_step,
            config_sha256=config_sha256,
            mode=mode,
            **extra,
        )

    def batch_committed(self, *, global_step, checkpoint_id, rank, microbatch_count,
                        packed_sample_ids, shard_ids, token_span_ids, loss_mask_hash,
                        position_policy, mixture_lane, curriculum_stage, opus_decision_id,
                        batch_hash, packing_utilization, loss_bearing_tokens, **extra) -> dict:
        return self.append(
            BATCH_COMMITTED,
            global_step=global_step,
            checkpoint_id=checkpoint_id,
            rank=rank,
            microbatch_count=microbatch_count,
            packed_sample_ids=list(packed_sample_ids),
            shard_ids=list(shard_ids),
            token_span_ids=list(token_span_ids),
            loss_mask_hash=loss_mask_hash,
            position_policy=position_policy,
            mixture_lane=mixture_lane,
            curriculum_stage=curriculum_stage,
            opus_decision_id=opus_decision_id,
            batch_hash=batch_hash,
            packing_utilization=packing_utilization,
            loss_bearing_tokens=loss_bearing_tokens,
            **extra,
        )

    def checkpoint_bound(self, *, global_step, checkpoint_id, dataloader_state,
                         rng_state="captured_where_available", **extra) -> dict:
        return self.append(
            CHECKPOINT_BOUND,
            global_step=global_step,
            checkpoint_id=checkpoint_id,
            model_state="saved",
            optimizer_state="saved",
            dataloader_state=dataloader_state,
            rng_state=rng_state,
            **extra,
        )

    def trainer_crashed(self, *, global_step, checkpoint_id, failed_rank, **extra) -> dict:
        return self.append(
            TRAINER_CRASHED,
            global_step=global_step,
            checkpoint_id=checkpoint_id,
            failed_rank=failed_rank,
            **extra,
        )

    def worker_crash_recovered(self, *, global_step, checkpoint_id, failed_rank,
                               next_expected_offset, recovery_mode="resume_from_last_committed_offset",
                               **extra) -> dict:
        return self.append(
            WORKER_CRASH_RECOVERED,
            global_step=global_step,
            checkpoint_id=checkpoint_id,
            failed_rank=failed_rank,
            recovery_mode=recovery_mode,
            next_expected_offset=next_expected_offset,
            **extra,
        )

    def checkpoint_restored(self, *, global_step, checkpoint_id, mode, **extra) -> dict:
        return self.append(
            CHECKPOINT_RESTORED,
            global_step=global_step,
            checkpoint_id=checkpoint_id,
            mode=mode,
            **extra,
        )


# ---------------------------------------------------------------------------
# Reading and verification
# ---------------------------------------------------------------------------


def read_events(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def batches(events: list[dict], branch_id: str | None = None) -> list[dict]:
    return [
        e
        for e in events
        if e["event"] == BATCH_COMMITTED
        and (branch_id is None or e["run_branch_id"] == branch_id)
    ]


def last_committed_step(events: list[dict], branch_id: str) -> int | None:
    committed = batches(events, branch_id)
    return max((e["global_step"] for e in committed), default=None)


def effective_batches(events: list[dict], branch_id: str) -> dict[int, dict]:
    """The batch that actually stands for each step, after rollbacks.

    A crash rolls the model back to the last checkpoint and re-consumes the steps since.
    Those earlier events are not deleted -- the ledger is append-only -- they are
    *superseded*, and the one with the highest `recovery_epoch` wins. Reconstructing the
    effective stream this way is what makes "no skipped or repeated batches" checkable
    against a ledger that legitimately contains a step more than once.
    """
    effective: dict[int, dict] = {}
    for event in batches(events, branch_id):
        step = event["global_step"]
        current = effective.get(step)
        if current is None or event.get("recovery_epoch", 0) >= current.get("recovery_epoch", 0):
            effective[step] = event
    return effective


def verify_append_only(events: list[dict]) -> dict:
    """Structural checks a consumption ledger must pass to be worth anything.

    Offsets must be contiguous from zero with no duplicates and no gaps -- that is the
    append-only guarantee. Then, per branch, the *effective* stream (highest recovery
    epoch per step) must cover a contiguous run of steps from zero with no holes. A gap
    means a batch was consumed without being recorded; a duplicate at the same recovery
    epoch means the same batch was trained on twice against the same model state.
    """
    offsets = [e["ledger_offset"] for e in events]
    expected = list(range(len(events)))
    duplicates = sorted({o for o in offsets if offsets.count(o) > 1})

    issues: list[str] = []
    branches = sorted({e["run_branch_id"] for e in events})
    per_branch: dict[str, dict] = {}

    for branch in branches:
        committed = batches(events, branch)
        if not committed:
            continue

        # Within one recovery epoch a step must never be committed twice.
        seen: set[tuple[int, int]] = set()
        for event in committed:
            key = (event.get("recovery_epoch", 0), event["global_step"])
            if key in seen:
                issues.append(
                    f"{branch}: step {key[1]} committed twice at recovery_epoch {key[0]}"
                )
            seen.add(key)

        effective = effective_batches(events, branch)
        steps = sorted(effective)
        start, end = steps[0], steps[-1]
        missing = sorted(set(range(start, end + 1)) - set(steps))
        if missing:
            issues.append(f"{branch}: effective stream missing steps {missing[:8]}")

        per_branch[branch] = {
            "events": len(committed),
            "effective_steps": len(steps),
            "superseded_by_rollback": len(committed) - len(steps),
            "step_range": [start, end],
            "recovery_epochs": sorted({e.get("recovery_epoch", 0) for e in committed}),
        }

    return {
        "events": len(events),
        "offsets_contiguous": offsets == expected,
        "duplicate_offsets": duplicates,
        "branches": branches,
        "per_branch": per_branch,
        "issues": issues,
        "ok": offsets == expected and not duplicates and not issues,
    }


def summarize(events: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for event in events:
        counts[event["event"]] = counts.get(event["event"], 0) + 1
    committed = batches(events)

    # Lane and token totals are computed over the *effective* stream, so the steps a
    # rollback superseded are not counted twice.
    effective: list[dict] = []
    for branch in sorted({e["run_branch_id"] for e in events}):
        effective.extend(effective_batches(events, branch).values())

    lanes: dict[str, int] = {}
    microbatch_lanes: dict[str, int] = {}
    for event in effective:
        lanes[event["mixture_lane"]] = lanes.get(event["mixture_lane"], 0) + 1
        for lane in event.get("microbatch_lanes", []):
            microbatch_lanes[lane] = microbatch_lanes.get(lane, 0) + 1

    return {
        "events": len(events),
        "by_event": counts,
        "batches_committed": len(committed),
        "effective_batches": len(effective),
        "superseded_by_rollback": len(committed) - len(effective),
        "steps_by_lane": lanes,
        "microbatches_by_lane": microbatch_lanes,
        "loss_bearing_tokens": sum(e["loss_bearing_tokens"] for e in effective),
        "branches": sorted({e["run_branch_id"] for e in events}),
        "verification": verify_append_only(events),
    }
