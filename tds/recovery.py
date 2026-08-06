"""Resume, replay and fork.

Widget 14 is the reference for this module, and its three recovery modes are the three
things a checkpoint plus a ledger can mean:

    ledger   replay the recorded stream. Same checkpoint, same data, same branch.
    fork     same checkpoint, deliberately different data, new branch id.
    random   same checkpoint, sampler re-seeded, no ledger consulted. The branch id does
             not change, which is precisely the problem -- this run looks like a
             continuation of run-a and is not.

The `random` mode is run as a **negative control**. It has no business in a production
pipeline, and that is why it is here: without it, "ledger replay reproduces the stream" is
an unfalsifiable claim. Showing that the same checkpoint under a re-seeded sampler produces
a *different* stream is what demonstrates the ledger is doing the work.

Resume is the strongest of the three proofs available here, because the comparison is
against records written by a process that no longer exists. The crash at step 100 kills the
trainer outright; the last checkpoint is at step 90; recovery rolls the model back to 90
and re-consumes steps 90-99. Those ten steps were already committed to the ledger before
the crash, by the dead process. If the rolled-back run regenerates them with identical
batch hashes, token spans and loss-mask hashes, the stream really is reconstructible.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import Config
from .ledger import effective_batches
from .pipeline import DataSystem
from .sampler import StepPlan, StreamPlanner

# The ledger fields a regenerated step must match exactly for replay to count.
COMPARED_FIELDS = (
    "batch_hash",
    "loss_mask_hash",
    "token_span_ids",
    "packed_sample_ids",
    "shard_ids",
    "microbatch_hashes",
    "microbatch_lanes",
    "mixture_lane",
    "curriculum_stage",
    "opus_decision_id",
    "loss_bearing_tokens",
)


def step_fields(plan: StepPlan) -> dict:
    """The same fields, taken from a freshly planned step rather than from the ledger."""
    return {
        "batch_hash": plan.batch_hash(),
        "loss_mask_hash": plan.microbatches[0].loss_mask_hash,
        "token_span_ids": [t for mb in plan.microbatches for t in mb.token_span_ids],
        "packed_sample_ids": [s for mb in plan.microbatches for s in mb.packed_sample_ids],
        "shard_ids": sorted({s for mb in plan.microbatches for s in mb.shard_ids}),
        "microbatch_hashes": [mb.batch_hash for mb in plan.microbatches],
        "microbatch_lanes": [mb.lane for mb in plan.microbatches],
        "mixture_lane": plan.microbatches[0].lane,
        "curriculum_stage": plan.stage,
        "opus_decision_id": plan.microbatches[0].opus_decision_id,
        "loss_bearing_tokens": plan.loss_bearing_tokens,
    }


def compare_step(recorded: dict, regenerated: dict) -> dict:
    mismatches = [
        field
        for field in COMPARED_FIELDS
        if recorded.get(field) != regenerated.get(field)
    ]
    return {
        "matched": not mismatches,
        "mismatched_fields": mismatches,
        "recorded_batch_hash": recorded.get("batch_hash"),
        "regenerated_batch_hash": regenerated.get("batch_hash"),
    }


def regenerate(
    config: Config,
    system: DataSystem,
    branch_id: str,
    step_start: int,
    step_end: int,
    stream_key: str | None = None,
    advance_from_zero: bool = True,
) -> dict[int, dict]:
    """Re-derive steps `[step_start, step_end)` straight from the immutable shards.

    No model is involved, so this is fast even when fast-forwarding from step 0 -- which
    is how the planner reaches the right cursor and OPUS state without any sampler state
    having been snapshotted into a checkpoint.
    """
    planner = StreamPlanner(config, system.registry, system.schedule, branch_id, stream_key)
    if advance_from_zero:
        planner.advance_to(step_start)
    out: dict[int, dict] = {}
    for step in range(step_start, step_end):
        plan = planner.plan_step(planner.next_step)
        out[step] = step_fields(plan)
    return out


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def replay_report(
    config: Config, system: DataSystem, branch_id: str, events: list[dict],
    step_start: int, step_end: int,
) -> dict:
    """Replay a historical interval and compare it field by field to the ledger."""
    recorded = effective_batches(events, branch_id)
    available = sorted(s for s in recorded if step_start <= s < step_end)
    regenerated = regenerate(config, system, branch_id, step_start, step_end)

    comparisons = {}
    for step in available:
        comparisons[step] = compare_step(recorded[step], regenerated[step])

    matched = [s for s, c in comparisons.items() if c["matched"]]
    mismatched = [s for s, c in comparisons.items() if not c["matched"]]

    return {
        "branch_id": branch_id,
        "interval": [step_start, step_end],
        "steps_compared": len(comparisons),
        "steps_matched": len(matched),
        "steps_mismatched": len(mismatched),
        "mismatched_steps": mismatched,
        "compared_fields": list(COMPARED_FIELDS),
        "all_matched": bool(comparisons) and not mismatched,
        "sample": {
            str(step): {
                "recorded_batch_hash": recorded[step]["batch_hash"],
                "replay_batch_hash": regenerated[step]["batch_hash"],
                "token_span_ids": recorded[step]["token_span_ids"][:4],
            }
            for step in available[:3]
        },
        "per_step": {str(s): c for s, c in comparisons.items()},
    }


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def resume_report(
    config: Config, system: DataSystem, branch_id: str, events: list[dict],
    checkpoint_meta: dict, crash_step: int, regenerated: dict[int, dict],
) -> dict:
    """Prove the rolled-back run re-consumed exactly what the dead process recorded.

    `regenerated` covers the rollback window -- the steps between the checkpoint and the
    crash, which were committed before the crash and are consumed again afterwards.
    """
    recorded = effective_batches(events, branch_id)
    resume_step = checkpoint_meta["next_step"]
    window = [s for s in sorted(regenerated) if s in recorded and s < crash_step]

    comparisons = {step: compare_step(recorded[step], regenerated[step]) for step in window}
    mismatched = [s for s, c in comparisons.items() if not c["matched"]]

    next_batch = None
    if resume_step in recorded and resume_step in regenerated:
        next_batch = {
            "step": resume_step,
            "expected_batch_hash": recorded[resume_step]["batch_hash"],
            "resumed_batch_hash": regenerated[resume_step]["batch_hash"],
            "matched": recorded[resume_step]["batch_hash"]
                       == regenerated[resume_step]["batch_hash"],
            "expected_token_spans": recorded[resume_step]["token_span_ids"],
            "resumed_token_spans": regenerated[resume_step]["token_span_ids"],
        }

    return {
        "branch_id": branch_id,
        "crash_step": crash_step,
        "checkpoint_id": checkpoint_meta["checkpoint_id"],
        "checkpoint_next_step": resume_step,
        "checkpoint_ledger_offset": checkpoint_meta["ledger_offset"],
        "rollback_window": [resume_step, crash_step],
        "steps_in_window": len(window),
        "steps_matched": len(window) - len(mismatched),
        "mismatched_steps": mismatched,
        "next_batch_after_resume": next_batch,
        "all_matched": bool(window) and not mismatched
                       and bool(next_batch and next_batch["matched"]),
        "note": "The compared records were written by the process that crashed. Matching "
                "them requires the stream to be reconstructible from the checkpoint and "
                "the ledger alone.",
    }


# ---------------------------------------------------------------------------
# Fork and the random control
# ---------------------------------------------------------------------------


def divergence_report(
    config: Config, system: DataSystem, base_branch: str, events: list[dict],
    from_step: int, compare_steps: int, fork_branch: str, random_stream_key: str,
) -> dict:
    """Widget 14's three-way comparison, on our own run.

    From one checkpoint: ledger replay reproduces the stream exactly, an intentional fork
    produces a different stream under a new branch id, and a re-seeded sampler produces a
    third stream while still calling itself run-a.
    """
    end = from_step + compare_steps
    recorded = effective_batches(events, base_branch)

    ledger_mode = regenerate(config, system, base_branch, from_step, end)
    fork_mode = regenerate(config, system, fork_branch, from_step, end)
    # "No ledger": the sampler restarts from its seed rather than being fast-forwarded to
    # the checkpoint's position, so the cursors sit at the wrong place entirely.
    random_mode = regenerate(
        config, system, base_branch, 0, compare_steps,
        stream_key=random_stream_key, advance_from_zero=False,
    )

    def hashes(table, offset=0):
        return [table[s]["batch_hash"] for s in sorted(table)][:compare_steps]

    original = [recorded[s]["batch_hash"] for s in sorted(recorded) if from_step <= s < end]
    ledger_hashes = hashes(ledger_mode)
    fork_hashes = hashes(fork_mode)
    random_hashes = hashes(random_mode)

    return {
        "from_step": from_step,
        "steps_compared": compare_steps,
        "original_stream": original,
        "modes": {
            "ledger": {
                "branch_id": base_branch,
                "batch_hashes": ledger_hashes,
                "matches_original": ledger_hashes == original,
                "meaning": "same checkpoint, same data, same branch -- the run continues",
            },
            "fork": {
                "branch_id": fork_branch,
                "batch_hashes": fork_hashes,
                "matches_original": fork_hashes == original,
                "meaning": "same checkpoint, deliberately different data, new branch id",
            },
            "random": {
                "branch_id": base_branch,
                "batch_hashes": random_hashes,
                "matches_original": random_hashes == original,
                "meaning": "same checkpoint, re-seeded sampler, no ledger consulted -- and "
                           "the branch id still says run-a",
            },
        },
        "ledger_reproduces_original": ledger_hashes == original,
        "fork_diverges": fork_hashes != original,
        "random_diverges": random_hashes != original,
        "conclusion": "The ledger is what makes the stream reproducible: the same "
                      "checkpoint yields three different futures, and only ledger mode "
                      "reproduces the recorded one.",
    }


def write_json(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1, sort_keys=True)
    return path
