"""Assembling the data system.

Every process in the demo -- the first training run, the resumed run after the crash, the
replay, the fork, the audit -- needs the same shard registry, the same mixture schedule and
the same vocabulary projection. They must agree exactly, because if the resuming process
built a *slightly* different registry then a matching batch hash would prove nothing.

So construction lives in one place and is deterministic. Shards are built once and loaded
thereafter: a second process finding shards already on disk reads them rather than
rebuilding, which is also what keeps the immutability guarantee honest. If a rebuild would
have produced different bytes, the content hashes recorded in the manifests would catch it.
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .corpus import build_corpus, corpus_summary
from .firewall import EvalFirewall, firewall_report
from .manifest import ShardRegistry, build_registry
from .mixture import MixtureSchedule
from .packing import PackDoc, compare_policies
from .shards import ShardWriter, load_shard
from .tokenizer import FrozenTokenizer
from .vocab import VocabProjection, build_projection


@dataclass
class DataSystem:
    config: Config
    tokenizer: FrozenTokenizer
    registry: ShardRegistry
    schedule: MixtureSchedule
    projection: VocabProjection
    vocab_coverage: float
    firewall: EvalFirewall
    reports: dict


def _shard_ids(shard_dir: Path) -> list[str]:
    return sorted(
        Path(p).name[: -len(".tokens.bin")]
        for p in glob.glob(str(Path(shard_dir) / "*.tokens.bin"))
    )


def build(config: Config, log=None, rebuild: bool = False) -> DataSystem:
    log = log or (lambda message: None)
    tokenizer = FrozenTokenizer(config)
    shard_dir = config.shard_dir
    existing = _shard_ids(shard_dir)
    reports: dict = {}

    if existing and not rebuild:
        shards = [load_shard(shard_dir, shard_id) for shard_id in existing]
        log(f"loaded {len(shards)} existing shards from {shard_dir}")
    else:
        documents = build_corpus(config)
        reports["corpus"] = corpus_summary(documents)
        log(f"corpus: {len(documents)} candidate documents across {len(config.lanes)} lanes")

        firewall = EvalFirewall(config)
        admitted, verdicts = firewall.scan_all(documents)
        reports["firewall"] = firewall_report(verdicts, firewall)
        blocked = reports["firewall"]["blocked"]
        log(f"eval firewall: {blocked} of {len(documents)} documents blocked, "
            f"{len(reports['firewall']['blocked_despite_trainable_flag'])} of them despite a "
            "trainable registry flag")

        shards = ShardWriter(config, tokenizer).write_all(admitted)
        log(f"shards created: {len(shards)} immutable shards, "
            f"{sum(s.token_count for s in shards)} tokens")

    registry = build_registry(shards, tokenizer, config)
    schedule = MixtureSchedule(config, registry)

    projection, coverage = build_projection(
        registry,
        size=config.require("model.projected_vocab_size"),
        special_ids=[tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id],
        source_vocab_size=tokenizer.vocab_size,
        unk_id=tokenizer.unk_id,
    )

    return DataSystem(
        config=config,
        tokenizer=tokenizer,
        registry=registry,
        schedule=schedule,
        projection=projection,
        vocab_coverage=coverage,
        firewall=EvalFirewall(config) if "firewall" not in reports else None,
        reports=reports,
    )


def packing_report(system: DataSystem, docs_per_lane: int = 40) -> dict:
    """Widget 5's comparison, run on our own documents in each lane.

    Every policy is measured on the same document set per lane, so the utilisation and
    boundary-risk trade-off is a result rather than a quotation.
    """
    context = system.config.require("batch.sequence_length")
    pad_id = system.tokenizer.pad_id
    out: dict = {"context_length": context, "by_lane": {}}

    for lane in system.config.lanes:
        entries = system.registry.admitted_by_lane(lane)
        docs: list[PackDoc] = []
        for entry in entries:
            for index, span in enumerate(entry.shard.doc_spans):
                tokens = entry.shard.span_tokens(span.token_start, span.token_end)
                docs.append(
                    PackDoc(
                        doc_id=span.doc_id,
                        shard_id=entry.shard_id,
                        lane=lane,
                        token_start=span.token_start,
                        token_ids=[int(t) for t in tokens],
                        segments=span.segments,
                    )
                )
                if len(docs) >= docs_per_lane:
                    break
            if len(docs) >= docs_per_lane:
                break
        if not docs:
            continue
        out["by_lane"][lane] = {
            "documents": len(docs),
            "configured_policy": system.config.lane(lane)["packing_policy"],
            "policies": compare_policies(docs, context, pad_id),
        }
    return out


def write_reports(system: DataSystem, manifests_dir: Path, ledgers_dir: Path) -> dict:
    manifests_dir = Path(manifests_dir)
    ledgers_dir = Path(ledgers_dir)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    ledgers_dir.mkdir(parents=True, exist_ok=True)

    written = {"shard_index": str(system.registry.write(manifests_dir))}
    written["mixture_schedule"] = str(
        system.schedule.write(manifests_dir / "mixture_schedule.json", system.registry)
    )

    if "firewall" in system.reports:
        path = ledgers_dir / "firewall_report.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(system.reports["firewall"], fh, ensure_ascii=False, indent=1, sort_keys=True)
        written["firewall_report"] = str(path)

    if "corpus" in system.reports:
        path = manifests_dir / "corpus_summary.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(system.reports["corpus"], fh, ensure_ascii=False, indent=1, sort_keys=True)
        written["corpus_summary"] = str(path)

    report = packing_report(system)
    path = manifests_dir / "packing_report.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1, sort_keys=True)
    written["packing_report"] = str(path)

    path = manifests_dir / "vocab_projection.json"
    system.projection.write(path, system.vocab_coverage)
    written["vocab_projection"] = str(path)

    return written
