"""Throughput and packing efficiency.

Widget 15's point is that raw tokens per second is a vanity metric. A loader can report a
healthy number while most of what it moved never trained anything: padding, context-only
positions in supervised samples, and batches that OPUS prepared and then threw away. So
every token the pipeline touched is assigned to exactly one of four buckets, and the
headline figure is **useful loss-bearing tokens per second**, not tokens per second.

    useful          loss-bearing positions the model was actually scored on
    opus_rejected   positions prepared, packed and masked, then rejected by the selector
    padding_waste   pad positions plus context-only positions carrying no loss
    loader_wait     GPU-equivalent time lost preparing data rather than computing

Every number is measured. `data_seconds` and `compute_seconds` come from `perf_counter`
around the two phases of the training loop; the token counts come from the batches
themselves and are recomputable from the ledger. The audit recomputes them and compares.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_report(
    trainer_summary: dict, ledger_summary: dict, schedule, registry,
    vocab_coverage: float, packing_report: dict,
) -> dict:
    timings = trainer_summary["timings"]
    planner = trainer_summary["planner"]

    useful = ledger_summary["loss_bearing_tokens"]
    prepared = planner["positions_prepared"]
    rejected = planner["positions_rejected"]

    # Every position the loader prepared lands in exactly one bucket. Positions inside
    # accepted batches that carry no loss -- padding, and context-only turns in supervised
    # samples -- are the waste widget 15 colours red.
    padding_waste = max(prepared - rejected - useful, 0)
    total = useful + rejected + padding_waste

    compute = timings["compute_seconds"]
    data = timings["data_seconds"]
    wall = compute + data

    def share(value):
        return round(value / total, 4) if total else 0.0

    return {
        "_doc": "Token fate and throughput, all measured. Recomputable from the "
                "consumption ledger and the trainer's own timings.",
        "token_fate": {
            "useful_loss_bearing": useful,
            "opus_rejected": rejected,
            "padding_and_context_waste": padding_waste,
            "total_prepared": total,
            "shares": {
                "useful": share(useful),
                "opus_rejected": share(rejected),
                "padding_and_context_waste": share(padding_waste),
            },
        },
        "throughput": {
            "useful_tokens_per_second": round(useful / wall, 1) if wall else 0.0,
            "prepared_tokens_per_second": round(total / wall, 1) if wall else 0.0,
            "compute_seconds": compute,
            "data_seconds": data,
            "wall_seconds": round(wall, 4),
            "loader_wait_share": round(data / wall, 4) if wall else 0.0,
            "steps": trainer_summary["steps_trained"],
            "seconds_per_step": round(wall / trainer_summary["steps_trained"], 4)
            if trainer_summary["steps_trained"] else 0.0,
        },
        "packing": {
            "configured_policy_by_lane": {
                lane: report["configured_policy"]
                for lane, report in sorted(packing_report.get("by_lane", {}).items())
            },
            "utilization_by_lane": {
                lane: report["policies"][report["configured_policy"]]["utilization"]
                for lane, report in sorted(packing_report.get("by_lane", {}).items())
            },
            "context_length": packing_report.get("context_length"),
        },
        "opus": {
            "decisions": trainer_summary["opus"]["decisions"],
            "rejection_rate": trainer_summary["opus"]["rejection_rate"],
            "planned_keep_fraction": schedule.keep_fraction,
            "candidates_prepared": planner["candidates_prepared"],
            "slots_hitting_attempt_cap": planner["slots_hitting_attempt_cap"],
        },
        "supply": {
            "lane_epochs": planner["lane_epochs"],
            "epoch_ceiling": schedule.epoch_ceiling,
            "documents_drawn": planner["documents_drawn"],
            "admitted_shards": len(registry.admitted),
        },
        "model": {
            "parameters": trainer_summary["parameters"],
            "vocab_projection_coverage": vocab_coverage,
        },
        "caveat": "Wall time is CPU NumPy on a toy model. The ratios -- what share of "
                  "prepared tokens became loss-bearing, how much went to padding, how "
                  "much OPUS discarded -- are the transferable numbers here, not the "
                  "absolute rate.",
    }


def mixture_compliance(schedule, ledger_summary: dict, trainer_summary: dict) -> dict:
    """Planned versus actual lane shares, measured two ways.

    The two measures disagree, and the disagreement is the interesting part. The schedule
    allocates *microbatch slots*; the model learns from *loss-bearing tokens*. A lane like
    agentic, where user turns and tool observations are masked, delivers far fewer scored
    tokens per slot than a lane like web where every position counts. Reporting only the
    slot share would hide that, and reporting only the token share would look like the
    scheduler had missed its target when it hit it exactly.
    """
    planned = schedule.planned_lane_shares()
    slots = ledger_summary.get("microbatches_by_lane", {})
    slot_total = sum(slots.values()) or 1
    actual_slots = {lane: round(count / slot_total, 4) for lane, count in sorted(slots.items())}
    actual_tokens = trainer_summary.get("consumed_lane_shares", {})

    rows = {}
    for lane in sorted(set(planned) | set(actual_slots) | set(actual_tokens)):
        planned_share = planned.get(lane, 0.0)
        slot_share = actual_slots.get(lane, 0.0)
        rows[lane] = {
            "planned_slot_share": round(planned_share, 4),
            "actual_slot_share": slot_share,
            "slot_drift": round(slot_share - planned_share, 4),
            "actual_loss_bearing_token_share": actual_tokens.get(lane, 0.0),
        }

    worst = max((abs(r["slot_drift"]) for r in rows.values()), default=0.0)
    return {
        "_doc": "Planned versus actual. Slot share is what the compiled schedule promised; "
                "token share is what the model was actually scored on.",
        "by_lane": rows,
        "max_absolute_slot_drift": round(worst, 4),
        "slot_shares_match_plan": worst <= 0.02,
        "note": "Slot drift comes only from OPUS retries within a slot's lane, so it stays "
                "near zero by construction. Token share differs from slot share because "
                "lanes differ in how many of their positions carry loss.",
    }


def write(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1, sort_keys=True)
    return path
