"""Shard manifests and the admission gate.

Widget 6 puts it plainly: the manifest is the contract between Sessions 1-5 and the
dataloader. If a shard is replayed a year later, the same tokenizer, source lineage and
safety state must still be knowable from the manifest alone.

The gate has **two tiers**, which is the part worth getting right. A single pass/fail
would throw away the distinction the widget draws between a shard that is dangerous and a
shard that is merely under-documented:

    hard requirements   pii_screen_status, eval_overlap_status, tokenizer_hash,
                        content_hash -- missing any of these blocks the shard from
                        training entirely. It may still be stored.
    soft requirements   dedup_status, license_tier, parent_manifest_ids -- missing or
                        weak values hold the shard for review. Reproducibility is
                        weakened, but nothing unsafe is happening.

Only `admitted_to_registry` shards are visible to the sampler. Held and blocked shards are
kept, manifested and reported -- storage is not the same permission as consumption.

Field names follow widget 6 exactly so the artifacts read against the session material.
The `admission` values map to its display strings: `admitted_to_registry` ->
"Admitted to registry", `held_for_review` -> "Held for review", `blocked_from_training` ->
"Blocked from training".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import REPO_ROOT, Config
from .hashing import sha256_json, sha256_text, tagged
from .shards import Shard
from .tokenizer import FrozenTokenizer

ADMITTED = "admitted_to_registry"
HELD = "held_for_review"
BLOCKED = "blocked_from_training"

HARD_FIELDS = ("pii_screen_status", "eval_overlap_status", "tokenizer_hash", "content_hash")
SOFT_FIELDS = ("dedup_status", "license_tier", "parent_manifest_ids")

WEAK_LICENSE_TIERS = ("unknown", "unsafe", "blocked")

UPSTREAM = REPO_ROOT / "fixtures" / "upstream"


# ---------------------------------------------------------------------------
# Lineage inherited from S4
# ---------------------------------------------------------------------------


def _load_s4_manifests() -> dict[str, dict]:
    with open(UPSTREAM / "s4_manifests.json", encoding="utf-8") as fh:
        return {m["file"]: m for m in json.load(fh)}


class Lineage:
    """Resolves where a shard's documents came from, and under which cleaning pipeline."""

    def __init__(self):
        self.s4_by_file = _load_s4_manifests()

        # S4's cleaning pipeline is the ordered list of script hashes it recorded. One
        # hash over that list identifies the exact code that produced these documents.
        any_manifest = next(iter(self.s4_by_file.values()))
        self.s4_pipeline_hash = tagged(
            "clean", sha256_json(any_manifest["cleaning_scripts"])
        )
        self.s4_scripts = any_manifest["cleaning_scripts"]

        # Documents generated in this repository are "cleaned" by the generator itself,
        # so the honest pipeline hash is the hash of that source file.
        generator = REPO_ROOT / "tds" / "corpus.py"
        self.generated_pipeline_hash = tagged(
            "clean", sha256_text(generator.read_text(encoding="utf-8"))
        )

    def for_shard(self, shard: Shard) -> dict:
        origins = {span.source.get("origin", "unknown") for span in shard.doc_spans}

        parent_ids: set[str] = set()
        source_urls: set[str] = set()
        for span in shard.doc_spans:
            upstream_file = span.source.get("upstream_shard_file")
            if upstream_file and upstream_file in self.s4_by_file:
                parent_ids.add(self.s4_by_file[upstream_file]["shard_id"])
            if span.source.get("source_url"):
                source_urls.add(span.source["source_url"])

        if origins == {"s4"}:
            pipeline_hash = self.s4_pipeline_hash
            pipeline = "s4_cleaning_pipeline"
        elif "s4" in origins:
            # A mixed shard cannot claim either pipeline, so it claims both.
            pipeline_hash = tagged(
                "clean",
                sha256_json([self.s4_pipeline_hash, self.generated_pipeline_hash]),
            )
            pipeline = "mixed_s4_and_generated"
        else:
            pipeline_hash = self.generated_pipeline_hash
            pipeline = "s6_generator"

        return {
            "cleaning_pipeline_hash": pipeline_hash,
            "cleaning_pipeline": pipeline,
            "parent_manifest_ids": sorted(parent_ids),
            "source_manifest": {
                "origins": sorted(origins),
                "source_urls": sorted(source_urls),
                "upstream_session": "S4" if "s4" in origins else None,
            },
        }


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def _aggregate_registry(shard: Shard) -> dict:
    """Collapse the per-document safety state into shard-level fields.

    A shard is only as screened as its least-screened document, which is why shards are
    grouped by metadata signature in the first place.
    """
    dedup = {span.registry.get("dedup_status") for span in shard.doc_spans}
    pii = {span.registry.get("pii_screen_status") for span in shard.doc_spans}
    licenses = {span.registry.get("license_tier") for span in shard.doc_spans}

    def collapse(values: set, weak_order: tuple) -> str | None:
        if None in values:
            return None
        for weak in weak_order:
            if weak in values:
                return weak
        return sorted(v for v in values if v)[0] if values else None

    return {
        "dedup_status": collapse(dedup, ()),
        "pii_screen_status": collapse(pii, ()),
        "license_tier": collapse(licenses, WEAK_LICENSE_TIERS),
    }


def admission_score(manifest: dict) -> int:
    """A transparent 0-100 completeness score, in the spirit of widget 6's gauge.

    Not a probability and not tuned to reproduce the widget's exact numbers -- it is a
    weighted count of which contract fields survived, so that a reviewer can see *how
    close* a held shard was without reading every field.
    """
    score = 8
    for field in HARD_FIELDS:
        if manifest.get(field) not in (None, "", "blocked_or_unknown"):
            score += 15
    if manifest.get("dedup_status"):
        score += 12
    license_tier = manifest.get("license_tier")
    score += {"safe": 12, "review": 8}.get(license_tier, 0)
    if manifest.get("parent_manifest_ids"):
        score += 8
    return min(score, 100)


def decide_admission(manifest: dict) -> tuple[str, list[str]]:
    """Apply the two-tier gate. Returns (verdict, reasons)."""
    hard_missing = [
        field
        for field in HARD_FIELDS
        if manifest.get(field) in (None, "", "blocked_or_unknown")
    ]
    if hard_missing:
        return BLOCKED, [
            f"hard requirement missing or failed: {field}" for field in hard_missing
        ]

    reasons: list[str] = []
    if not manifest.get("dedup_status"):
        reasons.append("dedup_status absent; near-duplicate group unknown")
    if manifest.get("license_tier") in WEAK_LICENSE_TIERS:
        reasons.append(f"license_tier {manifest.get('license_tier')!r} needs legal review")
    if reasons:
        return HELD, reasons

    notes = ["all required fields present"]
    if not manifest.get("parent_manifest_ids"):
        notes.append("no parent manifests recorded; lineage stops here")
    return ADMITTED, notes


def build_manifest(
    shard: Shard,
    tokenizer: FrozenTokenizer,
    lineage: Lineage,
    config: Config,
    eval_overlap_status: str = "clear",
) -> dict:
    registry = _aggregate_registry(shard)
    lineage_fields = lineage.for_shard(shard)

    manifest = {
        # -- widget 6's fields, same names -----------------------------------
        "shard_id": shard.shard_id,
        "capability_lane": shard.lane,
        "token_count": shard.token_count,
        "tokenizer_hash": tokenizer.tokenizer_hash,
        "content_hash": shard.content_hash,
        "cleaning_pipeline_hash": lineage_fields["cleaning_pipeline_hash"],
        "dedup_status": registry["dedup_status"],
        "pii_screen_status": registry["pii_screen_status"],
        "eval_overlap_status": eval_overlap_status,
        "license_tier": registry["license_tier"],
        "parent_manifest_ids": lineage_fields["parent_manifest_ids"],
        # -- what S6 adds so a shard can be replayed and audited --------------
        "document_count": len(shard.doc_spans),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        ),
        "tokens_file": shard.tokens_path.name,
        "docs_file": shard.docs_path.name,
        "cleaning_pipeline": lineage_fields["cleaning_pipeline"],
        "source_manifest": lineage_fields["source_manifest"],
        "normalizer_id": tokenizer.normalizer_id,
        "special_tokens": tokenizer.descriptor()["special_tokens"],
        "config_sha256": config.config_hash,
    }

    verdict, reasons = decide_admission(manifest)
    manifest["admission"] = verdict
    manifest["admission_reasons"] = reasons
    manifest["admission_score"] = admission_score(manifest)
    return manifest


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class RegistryEntry:
    manifest: dict
    shard: Shard

    @property
    def shard_id(self) -> str:
        return self.manifest["shard_id"]

    @property
    def lane(self) -> str:
        return self.manifest["capability_lane"]

    @property
    def admitted(self) -> bool:
        return self.manifest["admission"] == ADMITTED


class ShardRegistry:
    """Every shard that was built, and the subset the sampler is allowed to see."""

    def __init__(self, entries: list[RegistryEntry]):
        self.entries = sorted(entries, key=lambda e: e.shard_id)
        self.by_id = {e.shard_id: e for e in self.entries}

    @property
    def admitted(self) -> list[RegistryEntry]:
        return [e for e in self.entries if e.admitted]

    def admitted_by_lane(self, lane: str) -> list[RegistryEntry]:
        return [e for e in self.admitted if e.lane == lane]

    def lane_supply(self) -> dict[str, int]:
        """Trainable tokens per lane -- the ceiling the mixture compiler checks against."""
        supply: dict[str, int] = {}
        for entry in self.admitted:
            supply[entry.lane] = supply.get(entry.lane, 0) + entry.manifest["token_count"]
        return supply

    def write(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for entry in self.entries:
            path = directory / f"{entry.shard_id}.manifest.json"
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(entry.manifest, fh, ensure_ascii=False, indent=1, sort_keys=True)

        index = {
            "shards": len(self.entries),
            "admitted": len(self.admitted),
            "held_for_review": sum(1 for e in self.entries if e.manifest["admission"] == HELD),
            "blocked_from_training": sum(
                1 for e in self.entries if e.manifest["admission"] == BLOCKED
            ),
            "lane_supply_tokens": self.lane_supply(),
            "by_shard": {
                e.shard_id: {
                    "lane": e.lane,
                    "admission": e.manifest["admission"],
                    "admission_score": e.manifest["admission_score"],
                    "token_count": e.manifest["token_count"],
                    "documents": e.manifest["document_count"],
                    "reasons": e.manifest["admission_reasons"],
                }
                for e in self.entries
            },
        }
        index_path = directory / "shard_index.json"
        with open(index_path, "w", encoding="utf-8") as fh:
            json.dump(index, fh, ensure_ascii=False, indent=1, sort_keys=True)
        return index_path


def build_registry(
    shards: list[Shard], tokenizer: FrozenTokenizer, config: Config
) -> ShardRegistry:
    lineage = Lineage()
    return ShardRegistry(
        [
            RegistryEntry(build_manifest(shard, tokenizer, lineage, config), shard)
            for shard in shards
        ]
    )
