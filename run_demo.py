#!/usr/bin/env python3
"""One command. The complete Training Data Execution System demonstration.

    python run_demo.py

Regenerates `submission_artifacts/` from nothing but the committed fixtures and
`config/run_config.json`. No network, no manual steps.

The demo runs its training phases as **subprocesses**, which is not incidental. The crash
at step 100 is a real `os._exit(1)`: the process dies with its model, optimizer, sampler
cursors and OPUS state in memory, and none of it survives. The resuming process is a fresh
interpreter that has to rebuild the entire data stream from the checkpoint and the ledger
on disk. Had the whole demo run in one process, "the next batch is the expected batch"
would be a tautology -- the same object compared to itself.

Phases:

    build     corpus -> firewall -> shards -> manifests -> mixture schedule
    train-a   train run-a until it deliberately crashes mid-interval
    resume-a  roll back to the last checkpoint, re-consume, run to completion
    fork      branch run-b from an earlier checkpoint
    analyze   replay, divergence control, learning ledger, performance, audit, evidence
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from tds import audit as audit_mod
from tds import evidence as evidence_mod
from tds import learning as learning_mod
from tds import perf as perf_mod
from tds import pipeline, recovery
from tds.checkpoint import CheckpointStore
from tds.config import REPO_ROOT, Config
from tds.ledger import (MODE_FORK, MODE_LEDGER, MODE_RANDOM, ConsumptionLedger,
                        read_events, summarize)
from tds.trainer import Trainer

CRASH_EXIT_CODE = 17


class RunLog:
    """The execution log. Written to both the console and submission_artifacts/run.log."""

    def __init__(self, path: Path, echo: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo

    def __call__(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        if self.echo:
            print(line, flush=True)
        # The orchestrator clears submission_artifacts/ before starting, so the directory
        # has to be re-created rather than assumed.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()

    def event(self, name: str, detail: str = "") -> None:
        self(f"=== {name}{': ' + detail if detail else ''}")

    def reset(self) -> None:
        if self.path.exists():
            self.path.unlink()


def state_path(config: Config, name: str) -> Path:
    path = config.work_dir / "state" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_state(config: Config, name: str, payload: dict) -> None:
    with open(state_path(config, name), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, sort_keys=True)


def load_state(config: Config, name: str) -> dict:
    with open(state_path(config, name), encoding="utf-8") as fh:
        return json.load(fh)


def artifacts(config: Config) -> dict[str, Path]:
    root = config.artifacts_dir
    return {
        "root": root,
        "manifests": root / "manifests",
        "ledgers": root / "ledgers",
        "checkpoints": root / "checkpoints",
        "reports": root / "reports",
        "run_log": root / "run.log",
    }


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def phase_build(config: Config, log: RunLog) -> None:
    paths = artifacts(config)
    log.event("shards created")
    system = pipeline.build(config, log=log)

    log.event("manifests validated")
    written = pipeline.write_reports(system, paths["manifests"], paths["ledgers"])

    tokenizer_ok = system.tokenizer.verify(system.tokenizer.tokenizer_hash)
    manifests_agree = all(
        entry.manifest["tokenizer_hash"] == system.tokenizer.tokenizer_hash
        for entry in system.registry.entries
    )
    shards_intact = all(entry.shard.verify_content_hash() for entry in system.registry.entries)
    if tokenizer_ok and manifests_agree and shards_intact:
        log(f"[PASS] tokenizer_hash_verified {system.tokenizer.tokenizer_hash} "
            f"across {len(system.registry.entries)} manifests; all shard content hashes match")
    else:
        log("[FAIL] tokenizer_hash_verified")

    index = json.loads((paths["manifests"] / "shard_index.json").read_text(encoding="utf-8"))
    log(f"admission gate: {index['admitted']} admitted, {index['held_for_review']} held for "
        f"review, {index['blocked_from_training']} blocked from training")

    log.event("evaluation data blocked")
    report = system.reports["firewall"]
    log(f"eval firewall scanned {report['scanned']} documents against "
        f"{report['config']['fingerprints_total']} benchmark 13-grams")
    for record in report["blocked_documents"]:
        log(f"  blocked {record['doc_id']} ({record['lane']}) -- {record['reason']}")
    if report["blocked"] and report["blocked_despite_trainable_flag"]:
        log(f"[PASS] eval_shard_blocked {report['blocked']} documents blocked, "
            f"{len(report['blocked_despite_trainable_flag'])} of them despite a trainable "
            "registry flag")
    else:
        log("[FAIL] eval_shard_blocked")

    log.event("mixture compiled")
    supply = system.schedule.supply_check(system.registry)
    for lane, row in supply["by_lane"].items():
        log(f"  {lane:10s} planned {row['planned_tokens']:>7d}  after-OPUS "
            f"{row['required_after_opus']:>7d}  supply {row['verified_supply_tokens']:>7d}  "
            f"{row['status']} ({row['epochs_required']} epochs)")
    if supply["repetition_note"]:
        log(f"  {supply['repetition_note']}")
    if supply["compiler_warning"]:
        log(f"  {supply['compiler_warning']}")

    save_state(config, "build", {"written": {k: str(v) for k, v in written.items()},
                                 "vocab_coverage": system.vocab_coverage})


def _make_trainer(config: Config, system, branch_id: str, recovery_epoch: int, log: RunLog):
    paths = artifacts(config)
    ledger = ConsumptionLedger(paths["ledgers"] / "consumption.jsonl", branch_id)
    store = CheckpointStore(paths["checkpoints"])
    return ledger, store, Trainer(
        config, system.registry, system.schedule, system.projection,
        ledger, store, branch_id, recovery_epoch=recovery_epoch, log=log,
    )


def _dump_training_state(config: Config, name: str, trainer: Trainer) -> None:
    save_state(config, name, {
        "summary": trainer.summary(),
        "step_losses": trainer.step_losses,
        "attribution": {
            "by_shard": trainer.attribution.by_shard,
            "by_lane": trainer.attribution.by_lane,
        },
        "opus_records": [r.to_record() for r in trainer.planner.opus_ledger.records],
    })


def phase_train_a(config: Config, log: RunLog) -> None:
    system = pipeline.build(config)
    branch = config.require("run.primary_branch_id")
    crash_step = config.require("recovery.crash_at_step")

    ledger, store, trainer = _make_trainer(config, system, branch, 0, log)
    ledger.run_started(global_step=0, config_sha256=config.config_hash, mode=MODE_LEDGER,
                       recovery_epoch=0, total_steps=config.require("train.total_steps"))
    log.event("batches packed", f"training {branch} from step 0")
    log(f"model: {trainer.model.parameter_count} parameters, vocab projection "
        f"{system.projection.size} of {system.tokenizer.vocab_size}")

    log.event("crash simulated", f"scheduled at step {crash_step}")
    trainer.run(0, config.require("train.total_steps"), crash_at_step=crash_step)
    log("[FAIL] the run was supposed to crash and did not")


def phase_resume_a(config: Config, log: RunLog) -> None:
    paths = artifacts(config)
    system = pipeline.build(config)
    branch = config.require("run.primary_branch_id")
    crash_step = config.require("recovery.crash_at_step")

    events = read_events(paths["ledgers"] / "consumption.jsonl")
    store = CheckpointStore(paths["checkpoints"])
    meta = store.latest_for_branch(branch)
    if meta is None:
        raise RuntimeError("no checkpoint to resume from")

    log.event("run resumed")
    log(f"restoring {meta['checkpoint_id']} (step {meta['global_step']}, "
        f"{meta['dataloader_state']}); rolling back {crash_step - meta['next_step']} "
        "steps consumed after the checkpoint")

    ledger, store, trainer = _make_trainer(config, system, branch, 1, log)
    trainer.restore(meta["checkpoint_id"])
    ledger.checkpoint_restored(global_step=meta["global_step"],
                               checkpoint_id=meta["checkpoint_id"], mode=MODE_LEDGER,
                               recovery_epoch=1, next_step=meta["next_step"])
    ledger.worker_crash_recovered(
        global_step=crash_step, checkpoint_id=meta["checkpoint_id"],
        failed_rank=config.require("recovery.crashed_rank"),
        next_expected_offset=meta["ledger_offset"], recovery_epoch=1,
    )

    # Regenerate the rollback window before training it, so the comparison is against the
    # ledger the crashed process left behind rather than against anything this run wrote.
    regenerated = recovery.regenerate(config, system, branch, meta["next_step"], crash_step + 1)
    report = recovery.resume_report(config, system, branch, events, meta, crash_step, regenerated)
    recovery.write_json(paths["reports"] / "resume_report.json", report)

    next_batch = report["next_batch_after_resume"]
    if report["all_matched"] and next_batch and next_batch["matched"]:
        log(f"[PASS] resume_next_batch_matched step {next_batch['step']} "
            f"{next_batch['expected_batch_hash']} == {next_batch['resumed_batch_hash']}; "
            f"{report['steps_matched']}/{report['steps_in_window']} rollback-window steps "
            "identical to the pre-crash ledger")
    else:
        log(f"[FAIL] resume_next_batch_matched mismatched steps {report['mismatched_steps']}")

    total_steps = config.require("train.total_steps")
    trainer.run(meta["next_step"], total_steps)
    # The loop already checkpoints on interval boundaries; only save again if the run
    # ended between them, otherwise the final checkpoint would be written twice.
    if total_steps % config.require("train.checkpoint_interval") != 0:
        trainer.save_checkpoint(total_steps)
    _dump_training_state(config, "train-a", trainer)
    log(f"{branch} complete at step {config.require('train.total_steps')}, "
        f"final model hash {trainer.model.state_hash()}")


def phase_fork(config: Config, log: RunLog) -> None:
    paths = artifacts(config)
    system = pipeline.build(config)
    base = config.require("run.primary_branch_id")
    fork_branch = config.require("run.fork_branch_id")
    from_step = config.require("recovery.fork_from_step")
    fork_steps = config.require("recovery.fork_steps")

    store = CheckpointStore(paths["checkpoints"])
    parent = store.read_meta(f"ckpt_{from_step:05d}")

    log.event("branch forked", f"{base} -> {fork_branch} from step {from_step}")
    ledger, store, trainer = _make_trainer(config, system, fork_branch, 0, log)
    trainer.restore(parent["checkpoint_id"])
    ledger.checkpoint_restored(global_step=parent["global_step"],
                               checkpoint_id=parent["checkpoint_id"], mode=MODE_FORK,
                               recovery_epoch=0, parent_branch_id=base,
                               note="intentional fork: new branch id, same checkpoint")
    trainer.run(from_step, from_step + fork_steps)
    _dump_training_state(config, "fork", trainer)
    log(f"{fork_branch} trained {fork_steps} steps from {parent['checkpoint_id']}")


def phase_analyze(config: Config, log: RunLog) -> None:
    paths = artifacts(config)
    system = pipeline.build(config)
    branch = config.require("run.primary_branch_id")
    events = read_events(paths["ledgers"] / "consumption.jsonl")

    # --- replay -------------------------------------------------------------
    interval = config.require("recovery.replay_interval")
    log.event("historical stream replayed",
              f"steps {interval['step_start']}-{interval['step_end']}")
    report = recovery.replay_report(config, system, branch, events,
                                    interval["step_start"], interval["step_end"])
    recovery.write_json(paths["reports"] / "replay_report.json", report)
    if report["all_matched"]:
        log(f"[PASS] replay_hash_matched {report['steps_matched']}/{report['steps_compared']} "
            f"steps identical across {len(report['compared_fields'])} compared fields")
    else:
        log(f"[FAIL] replay_hash_matched mismatched {report['mismatched_steps']}")

    # --- three-way divergence control ---------------------------------------
    divergence = recovery.divergence_report(
        config, system, branch, events,
        from_step=config.require("recovery.random_control_from_step"),
        compare_steps=6,
        fork_branch=config.require("run.fork_branch_id"),
        random_stream_key=f"{branch}#reseeded-no-ledger",
    )
    recovery.write_json(paths["reports"] / "fork_report.json", divergence)
    log(f"recovery modes from step {divergence['from_step']}: "
        f"ledger reproduces={divergence['ledger_reproduces_original']}, "
        f"fork diverges={divergence['fork_diverges']}, "
        f"random diverges={divergence['random_diverges']}")

    # --- learning ledger ----------------------------------------------------
    train_state = load_state(config, "train-a")
    stage_order = [s["stage"] for s in config.require("mixture.stages")]

    learning = learning_mod.build_learning_ledger(
        train_state["attribution"], system.registry, train_state["opus_records"], stage_order
    )
    learning_mod.write(paths["ledgers"] / "learning_ledger.json", learning)
    log(f"learning ledger: {len(learning['by_shard'])} shards scored; lane hardness "
        f"{' > '.join(learning['lane_hardness_ranking'])}")
    for hint, shards in learning["policy_hints"].items():
        if shards:
            log(f"  {hint}: {len(shards)} shards")

    with open(paths["ledgers"] / "opus_decisions.jsonl", "w", encoding="utf-8") as fh:
        for record in train_state["opus_records"]:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    # --- performance --------------------------------------------------------
    log.event("performance measured")
    ledger_summary = summarize(events)
    packing = json.loads(
        (paths["manifests"] / "packing_report.json").read_text(encoding="utf-8")
    )
    performance = perf_mod.build_report(
        train_state["summary"], ledger_summary, system.schedule, system.registry,
        load_state(config, "build")["vocab_coverage"], packing,
    )
    compliance = perf_mod.mixture_compliance(system.schedule, ledger_summary,
                                             train_state["summary"])
    performance["mixture_compliance"] = compliance
    perf_mod.write(paths["root"] / "performance.json", performance)
    fate = performance["token_fate"]["shares"]
    log(f"token fate: useful {fate['useful']:.1%}, OPUS-rejected {fate['opus_rejected']:.1%}, "
        f"padding/context {fate['padding_and_context_waste']:.1%}; "
        f"{performance['throughput']['useful_tokens_per_second']} useful tokens/sec")

    # --- audit --------------------------------------------------------------
    # The index is written here rather than by the training phases, so it covers every
    # branch's checkpoints in one place regardless of which process created them.
    CheckpointStore(paths["checkpoints"]).write_index()

    log.event("audit completed")
    audit_report = audit_mod.run_audit(config, log=log)
    audit_mod.write(paths["reports"] / "audit_report.json", audit_report)
    log(f"audit: {audit_report['checks_passed']}/{audit_report['checks_total']} checks "
        f"passed against the generated artifacts alone")

    # --- evidence -----------------------------------------------------------
    evidence = evidence_mod.build(config, audit_report)
    evidence_mod.write(paths["root"], evidence)
    log(f"evidence bundle: {evidence['summary']['passed']}/{evidence['summary']['total']} "
        "requirements PASS")


PHASES = {
    "build": phase_build,
    "train-a": phase_train_a,
    "resume-a": phase_resume_a,
    "fork": phase_fork,
    "analyze": phase_analyze,
}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_subprocess(phase: str, log: RunLog, expect_exit: int | None = 0) -> None:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--phase", phase],
        cwd=str(REPO_ROOT),
    )
    if expect_exit is not None and result.returncode != expect_exit:
        raise SystemExit(
            f"phase {phase} exited {result.returncode}, expected {expect_exit}"
        )
    if expect_exit == CRASH_EXIT_CODE:
        log(f"trainer subprocess died with exit code {result.returncode} as intended; "
            "in-memory model, optimizer and sampler state are gone")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=sorted(PHASES), help=argparse.SUPPRESS)
    parser.add_argument("--config", default=None)
    parser.add_argument("--keep-work", action="store_true",
                        help="reuse an existing work/ directory instead of rebuilding")
    args = parser.parse_args()

    config = Config.load(args.config)
    paths = artifacts(config)
    log = RunLog(paths["run_log"])

    if args.phase:
        PHASES[args.phase](config, log)
        return

    # Fresh state: the demo must be reproducible from the fixtures alone.
    if not args.keep_work:
        import shutil
        for directory in (config.work_dir, config.artifacts_dir):
            shutil.rmtree(directory, ignore_errors=True)

    log.reset()
    started = time.perf_counter()
    log("S6 Training Data Execution System -- full demonstration")
    log(f"config {config.fingerprint()['config_path']} sha256 {config.config_hash}")

    phase_build(config, log)
    run_subprocess("train-a", log, expect_exit=CRASH_EXIT_CODE)
    run_subprocess("resume-a", log)
    run_subprocess("fork", log)
    phase_analyze(config, log)

    elapsed = time.perf_counter() - started
    log(f"demonstration complete in {elapsed:.1f}s -- artifacts in "
        f"{paths['root'].relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
