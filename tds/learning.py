"""The two-way learning ledger.

The consumption ledger records what the run ate. This one records what the run *learned
from what it ate*, and sends the answer back to data planning. Widget 12 gives it a shape:
per shard, a usefulness score, a loss delta, a hot-token share, an OPUS score, a loss trend
across model phases, and a `v6_policy_hint`.

The link between the two ledgers is explicit and joinable, as widget 12's backlinks show:

    batch_committed         a step consumed spans from this shard
    token_ppl_aggregated    per-token losses from those spans were aggregated
    learning_delta_attached the change between the shard's first and last exposure
    v6_policy_hint          what that implies for the next corpus

Every number here comes from `per_token` losses that the model actually produced. Nothing
is assumed from the lane a shard belongs to -- widget 11's finding that Indic stays the
hardest lane even late in training is treated as a *prediction to check*, not an input.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# Loss must improve by at least this much between a shard's first and last exposure
# before the shard counts as pulling its weight.
IMPROVEMENT_THRESHOLD = 0.05
STALLED_THRESHOLD = 0.01

KEEP = "keep_with_phase_guard"
MONITOR = "monitor_next_phase"
DELAY = "delay_or_reclean"


def _mean(entry: dict) -> float:
    return entry["loss_sum"] / entry["tokens"] if entry["tokens"] else 0.0


def _stage_trend(entry: dict, stage_order: list[str]) -> list[dict]:
    """Loss per curriculum stage, as an ordered list.

    A list rather than a dict on purpose: these artifacts are written with sorted keys, so
    a dict keyed by stage name would come back out alphabetically -- anneal, foundation,
    skill_build -- and a reader would draw the trend backwards.
    """
    return [
        {"stage": stage, "mean_loss": round(_mean(entry["by_stage"][stage]), 4)}
        for stage in stage_order
        if stage in entry["by_stage"] and entry["by_stage"][stage]["tokens"]
    ]


def _usefulness(loss_delta: float, tokens: int, max_tokens: int, opus_score: float) -> int:
    """A transparent 0-100 score, weighted the way widget 12's slider is by default.

    60% how much the loss moved, 20% how much of the run the shard actually carried, 20%
    what OPUS thought of it. Not a probability -- a ranking aid whose inputs are all
    visible in the same record.
    """
    improvement = max(min(-loss_delta / 1.5, 1.0), 0.0)
    volume = math.sqrt(tokens / max_tokens) if max_tokens else 0.0
    return int(round(100 * (0.6 * improvement + 0.2 * volume + 0.2 * opus_score)))


def _policy_hint(loss_delta: float, trend: list[dict]) -> tuple[str, str]:
    regressed = len(trend) >= 2 and trend[-1]["mean_loss"] > trend[-2]["mean_loss"]

    if regressed:
        return DELAY, ("Loss ticked back up in the final stage. Valuable earlier, weak "
                       "late; avoid spending anneal budget here without re-cleaning.")
    if loss_delta <= -IMPROVEMENT_THRESHOLD:
        return KEEP, ("Loss fell steadily across exposures. Keep, and guard the phase it "
                      "is scheduled into.")
    if loss_delta >= -STALLED_THRESHOLD:
        return DELAY, ("Loss barely moved across exposures. Either the content is already "
                       "learned or it is too noisy to learn from; re-clean or delay.")
    return MONITOR, "Modest improvement. Worth another phase before deciding."


def build_learning_ledger(
    attribution: dict, registry, opus_records: list[dict], stage_order: list[str],
    hot_threshold_ppl: float = 20.0,
) -> dict:
    """Join loss attribution against the shard registry and the OPUS board.

    Takes plain data -- `attribution` is `{"by_shard": ..., "by_lane": ...}` and
    `opus_records` is a list of decision-record dicts -- so it works identically whether
    those came from a live trainer or were read back off disk in a later process.
    """
    opus_by_shard: dict[str, list[float]] = {}
    for record in opus_records:
        for shard_id in record["shard_ids"]:
            opus_by_shard.setdefault(shard_id, []).append(record["score"])

    shards = attribution["by_shard"]
    max_tokens = max((e["tokens"] for e in shards.values()), default=1)

    per_shard = {}
    for shard_id, entry in sorted(shards.items()):
        trend = _stage_trend(entry, stage_order)
        loss_delta = (
            round(trend[-1]["mean_loss"] - trend[0]["mean_loss"], 4) if len(trend) >= 2 else 0.0
        )
        scores = opus_by_shard.get(shard_id, [])
        opus_score = round(sum(scores) / len(scores), 3) if scores else 0.0
        mean_loss = _mean(entry)
        hint, verdict = _policy_hint(loss_delta, trend)

        manifest = registry.by_id[shard_id].manifest if shard_id in registry.by_id else {}
        per_shard[shard_id] = {
            "shard_id": shard_id,
            "lane": entry["lane"],
            "tokens_scored": entry["tokens"],
            "mean_loss": round(mean_loss, 4),
            "mean_perplexity": round(math.exp(min(mean_loss, 20)), 2),
            "loss_delta": loss_delta,
            "loss_trend_by_stage": trend,
            "first_step": entry["first_step"],
            "last_step": entry["last_step"],
            "opus_score": opus_score,
            "opus_decisions": len(scores),
            "usefulness": _usefulness(loss_delta, entry["tokens"], max_tokens, opus_score),
            "v6_policy_hint": hint,
            "verdict": verdict,
            "manifest_token_count": manifest.get("token_count"),
            "ledger_backlinks": {
                "batch_committed": f"steps {entry['first_step']}-{entry['last_step']}",
                "token_ppl_aggregated": f"{entry['tokens']} scored tokens",
                "learning_delta_attached": loss_delta,
                "v6_policy_hint": hint,
            },
        }

    per_lane = {}
    for lane, entry in sorted(attribution["by_lane"].items()):
        trend = _stage_trend(entry, stage_order)
        mean_loss = _mean(entry)
        per_lane[lane] = {
            "lane": lane,
            "tokens_scored": entry["tokens"],
            "mean_loss": round(mean_loss, 4),
            "mean_perplexity": round(math.exp(min(mean_loss, 20)), 2),
            "loss_trend_by_stage": trend,
            "loss_delta": (
                round(trend[-1]["mean_loss"] - trend[0]["mean_loss"], 4)
                if len(trend) >= 2 else 0.0
            ),
        }

    ranking = sorted(per_lane.values(), key=lambda r: -r["mean_perplexity"])
    hardest = [r["lane"] for r in ranking]

    # Widget 11 found indic hardest early and still hardest late. Our corpus and model are
    # different, so this is recorded as a comparison rather than a confirmation.
    first_stage = stage_order[0]
    last_stage = stage_order[-1]

    def stage_ranking(stage):
        rows = []
        for lane, row in per_lane.items():
            for point in row["loss_trend_by_stage"]:
                if point["stage"] == stage:
                    rows.append((lane, point["mean_loss"]))
        return [lane for lane, _ in sorted(rows, key=lambda kv: -kv[1])]

    # A 180-step run does not reach every admitted shard: each lane's cursor walks its
    # shards in order and stops where the run stops. Reporting coverage keeps the
    # difference between "this shard was judged unhelpful" and "this shard was never
    # opened" visible.
    admitted = len(registry.admitted)
    coverage = {
        "admitted_shards": admitted,
        "shards_with_loss_attribution": len(per_shard),
        "shards_not_reached": admitted - len(per_shard),
        "note": "Shards not reached carry no verdict. A short run walks each lane's cursor "
                "only partway, so absence of a policy hint is not a negative judgement.",
    }

    return {
        "_doc": "Two-way learning ledger: per-token loss attributed back to the shards and "
                "lanes it came from, and what that implies for the next corpus.",
        "coverage": coverage,
        "hot_threshold_ppl": hot_threshold_ppl,
        "by_shard": per_shard,
        "by_lane": per_lane,
        "lane_hardness_ranking": hardest,
        "widget_11_comparison": {
            "widget_finding": "indic hardest both early and late (125.9 -> 17.3 avg ppl), "
                              "order indic > reasoning/agentic > code > pretrain",
            "measured_first_stage_ranking": stage_ranking(first_stage),
            "measured_last_stage_ranking": stage_ranking(last_stage),
            "note": "Different corpus, different tokenizer coverage and a 1.5M-parameter "
                    "model, so agreement is informative but disagreement is not a defect.",
        },
        "policy_hints": {
            hint: sorted(s for s, r in per_shard.items() if r["v6_policy_hint"] == hint)
            for hint in (KEEP, MONITOR, DELAY)
        },
    }


def write(path: Path, ledger: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(ledger, fh, ensure_ascii=False, indent=1, sort_keys=True)
    return path
