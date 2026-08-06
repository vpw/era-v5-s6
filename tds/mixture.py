"""The mixture schedule: S5's recipe compiled into something a dataloader can execute.

S5 produced a plan -- lane weights, protected floors, an anneal reserve, an OPUS keep
fraction. That plan is a document. This module turns it into a *schedule*: an explicit
lane assignment for every microbatch slot in the run, written out as an artifact before
training starts.

Compiling ahead of time rather than sampling lanes on the fly buys three things:

  * planned shares are exact, not approximately right after 1,200 draws;
  * `plan(branch, step, slot)` stays a pure function -- no counter has to be replayed to
    know which lane slot 900 belongs to, which is what makes replay and fork cheap;
  * the schedule is auditable. The audit re-reads it and checks the actual consumed
    shares against it, so "mixture compliance" is a comparison between two artifacts
    rather than a claim.

Slots are spread within a stage by stride scheduling (each lane's k-th occurrence is
placed at (k + 0.5) / n_lane through the stage, then all placements are sorted). Lanes
interleave evenly instead of arriving in blocks, which matters because a run that trains
200 consecutive Indic microbatches is not the same experiment as one that spreads them.

The supply check follows widget 7: required-after-OPUS is `planned / keep_fraction`, and
any lane whose verified supply falls short is reported with the compiler's warning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import REPO_ROOT, Config
from .manifest import ShardRegistry

S5_LEDGER = REPO_ROOT / "fixtures" / "upstream" / "s5_ledger.json"


def load_s5_plan() -> dict:
    """The inherited mixture plan. S5's floors and OPUS keep fraction are used as given."""
    with open(S5_LEDGER, encoding="utf-8") as fh:
        ledger = json.load(fh)
    return {
        "budget_tokens": ledger["budget_tokens"],
        "main_mixture_pct": ledger["main_mixture_pct"],
        "anneal_mixture_pct": ledger["anneal_mixture_pct"],
        "floors_pct": ledger["floors_pct"],
        "always_on_lane_pct": ledger["always_on_lane_pct"],
        "epoch_ceiling": ledger["epoch_ceiling"],
        "opus_keep_fraction": ledger["opus"]["keep_fraction"],
    }


@dataclass
class StagePlan:
    stage: str
    token_start: int
    token_end: int
    slot_start: int
    slot_end: int
    weights: dict[str, float]
    planned_tokens: dict[str, int]

    @property
    def token_span(self) -> int:
        return self.token_end - self.token_start

    @property
    def slots(self) -> int:
        return self.slot_end - self.slot_start


def _stride_interleave(counts: dict[str, int]) -> list[str]:
    """Spread each lane's slots evenly across the stage rather than in a block."""
    placements: list[tuple[float, str, int]] = []
    for lane in sorted(counts):
        n = counts[lane]
        for k in range(n):
            placements.append(((k + 0.5) / n, lane, k))
    placements.sort()
    return [lane for _, lane, _ in placements]


def _largest_remainder(weights: dict[str, float], total: int) -> dict[str, int]:
    """Apportion `total` whole slots across lanes, hitting the target exactly."""
    raw = {lane: weights[lane] * total for lane in sorted(weights)}
    counts = {lane: int(value) for lane, value in raw.items()}
    shortfall = total - sum(counts.values())
    remainders = sorted(
        ((raw[lane] - counts[lane], lane) for lane in raw), reverse=True
    )
    for _, lane in remainders[:shortfall]:
        counts[lane] += 1
    return counts


class MixtureSchedule:
    def __init__(self, config: Config, registry: ShardRegistry | None = None):
        self.config = config
        self.s5 = load_s5_plan()

        self.total_tokens = config.require("mixture.total_tokens")
        self.floors = config.require("mixture.protected_floors")
        self.anneal_reserve_lanes = config.require("mixture.anneal_reserve_lanes")
        self.warmup_band_pct = config.require("mixture.warmup_band_pct")
        self.keep_fraction = self.s5["opus_keep_fraction"]
        self.epoch_ceiling = self.s5["epoch_ceiling"]

        self.tokens_per_slot = config.require("batch.microbatch_size") * config.require(
            "batch.sequence_length"
        )
        self.slots_per_step = config.require("batch.microbatches_per_step")
        self.total_steps = config.require("train.total_steps")
        self.total_slots = self.total_steps * self.slots_per_step

        self.stages = self._compile_stages()
        self.lane_plan = self._compile_lane_plan()
        self.registry = registry

    # -- compilation -------------------------------------------------------------

    def _compile_stages(self) -> list[StagePlan]:
        stages: list[StagePlan] = []
        weights_by_stage = self.config.require("mixture.stage_weights")
        for spec in self.config.require("mixture.stages"):
            name = spec["stage"]
            weights = {k: v for k, v in weights_by_stage[name].items() if not k.startswith("_")}
            self._check_weights(name, weights)
            slot_start = spec["token_start"] // self.tokens_per_slot
            slot_end = spec["token_end"] // self.tokens_per_slot
            planned = {
                lane: int(round(weights[lane] * (spec["token_end"] - spec["token_start"])))
                for lane in sorted(weights)
            }
            stages.append(
                StagePlan(
                    stage=name,
                    token_start=spec["token_start"],
                    token_end=spec["token_end"],
                    slot_start=slot_start,
                    slot_end=slot_end,
                    weights=weights,
                    planned_tokens=planned,
                )
            )
        return stages

    def _check_weights(self, stage: str, weights: dict[str, float]) -> None:
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"stage {stage!r} lane weights sum to {total}, not 1.0")
        for lane, floor in self.floors.items():
            if weights.get(lane, 0.0) + 1e-12 < floor:
                raise ValueError(
                    f"stage {stage!r} gives lane {lane!r} a share of {weights.get(lane, 0.0):.3f}, "
                    f"below its protected floor of {floor:.3f}. Protected floors are a "
                    "constraint on the schedule, not a suggestion to OPUS."
                )

    def _compile_lane_plan(self) -> list[str]:
        plan: list[str] = []
        for stage in self.stages:
            counts = _largest_remainder(stage.weights, stage.slots)
            plan.extend(_stride_interleave(counts))
        # The last stage's slot_end is derived from the token budget; if rounding leaves
        # the plan a slot short of the configured step count, extend with the anneal
        # stage's dominant lane rather than silently running off the end.
        while len(plan) < self.total_slots:
            plan.append(max(self.stages[-1].weights, key=lambda k: self.stages[-1].weights[k]))
        return plan[: self.total_slots]

    # -- lookups used by the sampler ---------------------------------------------

    def slot_index(self, step: int, microbatch: int) -> int:
        return step * self.slots_per_step + microbatch

    def lane_for(self, step: int, microbatch: int) -> str:
        return self.lane_plan[self.slot_index(step, microbatch) % len(self.lane_plan)]

    def stage_for_step(self, step: int) -> str:
        token_position = step * self.slots_per_step * self.tokens_per_slot
        for stage in self.stages:
            if stage.token_start <= token_position < stage.token_end:
                return stage.stage
        return self.stages[-1].stage

    def planned_lane_shares(self) -> dict[str, float]:
        counts: dict[str, int] = {}
        for lane in self.lane_plan:
            counts[lane] = counts.get(lane, 0) + 1
        total = len(self.lane_plan)
        return {lane: counts[lane] / total for lane in sorted(counts)}

    # -- supply check ------------------------------------------------------------

    def supply_check(self, registry: ShardRegistry) -> dict:
        """Widget 7's table: planned, required after OPUS, verified supply, verdict."""
        supply = registry.lane_supply()
        planned: dict[str, int] = {}
        for stage in self.stages:
            for lane, tokens in stage.planned_tokens.items():
                planned[lane] = planned.get(lane, 0) + tokens

        # Three verdicts, not two. A lane whose unique supply falls short of demand is
        # not automatically a planning failure -- S5 already budgets for deliberate
        # repetition up to an epoch ceiling. What matters is whether covering the gap
        # needs *more epochs than the ceiling allows*, which is the only case that
        # forces the plan to change.
        rows = {}
        repeated = []
        shortfalls = []
        for lane in sorted(planned):
            required = planned[lane] / self.keep_fraction
            verified = supply.get(lane, 0)
            epochs = (required / verified) if verified else float("inf")
            within_ceiling = epochs <= self.epoch_ceiling if verified else False
            if verified >= required:
                status = "satisfied_by_unique_tokens"
            elif within_ceiling:
                status = "covered_by_repetition"
                repeated.append(lane)
            else:
                status = "shortfall"
                shortfalls.append(lane)
            rows[lane] = {
                "planned_tokens": planned[lane],
                "required_after_opus": int(round(required)),
                "verified_supply_tokens": verified,
                "epochs_required": round(epochs, 3) if verified else None,
                "within_epoch_ceiling": within_ceiling,
                "status": status,
            }

        warning = None
        if shortfalls:
            lanes = ", ".join(l.capitalize() for l in shortfalls)
            warning = (
                f"Compiler warning: {lanes} supply is not enough after OPUS rejection. "
                "Lower share, collect more, use repetition deliberately, or protect only "
                "the highest-value subset."
            )

        repetition_note = None
        if repeated:
            lanes = ", ".join(l.capitalize() for l in repeated)
            repetition_note = (
                f"{lanes} demand exceeds unique supply and is covered by deliberate "
                f"repetition within S5's epoch ceiling of {self.epoch_ceiling}."
            )

        return {
            "keep_fraction": self.keep_fraction,
            "opus_rejection_rate": round(1.0 - self.keep_fraction, 4),
            "epoch_ceiling": self.epoch_ceiling,
            "by_lane": rows,
            "shortfall_lanes": shortfalls,
            "repetition_lanes": repeated,
            "compiler_warning": warning,
            "repetition_note": repetition_note,
        }

    # -- artifact ----------------------------------------------------------------

    def to_document(self, registry: ShardRegistry | None = None) -> dict:
        doc = {
            "_doc": "Compiled mixture schedule. Lane assignment per microbatch slot is "
                    "fixed here, before training, and the audit checks consumed shares "
                    "against it.",
            "inherited_from": {
                "session": "S5",
                "file": str(S5_LEDGER.relative_to(REPO_ROOT)),
                "floors_pct": self.s5["floors_pct"],
                "always_on_lane_pct": self.s5["always_on_lane_pct"],
                "opus_keep_fraction": self.keep_fraction,
                "epoch_ceiling": self.epoch_ceiling,
            },
            "total_tokens": self.total_tokens,
            "tokens_per_slot": self.tokens_per_slot,
            "slots_per_step": self.slots_per_step,
            "total_slots": self.total_slots,
            "warmup_band_pct": self.warmup_band_pct,
            "protected_floors": self.floors,
            "anneal_reserve_lanes": self.anneal_reserve_lanes,
            "stages": [
                {
                    "stage": s.stage,
                    "token_start": s.token_start,
                    "token_end": s.token_end,
                    "slot_start": s.slot_start,
                    "slot_end": s.slot_end,
                    "weights": s.weights,
                    "planned_tokens": s.planned_tokens,
                    "slots": s.slots,
                }
                for s in self.stages
            ],
            "planned_lane_shares": self.planned_lane_shares(),
            "lane_plan": self.lane_plan,
        }
        if registry is not None:
            doc["supply_check"] = self.supply_check(registry)
        return doc

    def write(self, path: Path, registry: ShardRegistry | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_document(registry), fh, ensure_ascii=False, indent=1, sort_keys=True)
        return path
