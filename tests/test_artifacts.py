"""Tests against the generated artifacts.

These are the ones that would catch a bundle whose numbers had drifted from the files
backing them. They skip cleanly when `submission_artifacts/` is absent, so `pytest` works
on a fresh clone before `run_demo.py` has been run.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tds.hashing import sha256_file, tagged
from tds.ledger import effective_batches, read_events, verify_append_only
from tds.manifest import admission_score, decide_admission
from tds.shards import TOKEN_DTYPE, load_shard


def read(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def evidence(artifacts):
    return read(artifacts / "evidence.json")


@pytest.fixture(scope="module")
def audit_report(artifacts):
    return read(artifacts / "reports" / "audit_report.json")


@pytest.fixture(scope="module")
def events(artifacts):
    return read_events(artifacts / "ledgers" / "consumption.jsonl")


# ---------------------------------------------------------------------------
# The bundle
# ---------------------------------------------------------------------------


def test_required_tree_exists(artifacts):
    """Exactly the structure the assignment specifies."""
    for relative in ["run.log", "evidence.json", "evidence.md", "performance.json",
                     "manifests", "ledgers", "checkpoints"]:
        assert (artifacts / relative).exists(), f"missing {relative}"


def test_run_log_contains_every_required_event(artifacts):
    log = (artifacts / "run.log").read_text(encoding="utf-8")
    for event in ["shards created", "manifests validated", "evaluation data blocked",
                  "mixture compiled", "batches packed", "checkpoint_saved",
                  "crash simulated", "run resumed", "historical stream replayed",
                  "branch forked", "audit completed", "performance measured"]:
        assert event in log, f"run.log never mentions {event!r}"


def test_run_log_pass_markers(artifacts):
    log = (artifacts / "run.log").read_text(encoding="utf-8")
    for marker in ["[PASS] tokenizer_hash_verified", "[PASS] eval_shard_blocked",
                   "[PASS] checkpoint_saved", "[PASS] resume_next_batch_matched",
                   "[PASS] replay_hash_matched"]:
        assert marker in log, f"run.log missing {marker}"
    assert "[FAIL]" not in log, "the run of record contains a FAIL marker"


def test_evidence_rows_match_the_assignment(evidence):
    expected = ["Tokenizer integrity", "Evaluation firewall", "Packing correctness",
                "Mixture compliance", "OPUS audit trail", "Crash recovery", "Replay",
                "Learning trace", "Throughput"]
    assert [r["requirement"] for r in evidence["requirements"]] == expected


def test_every_evidence_path_resolves(artifacts, evidence):
    """A claim pointing at a file that does not exist is not evidence."""
    from tds.config import REPO_ROOT
    missing = [
        path
        for row in evidence["requirements"]
        for path in row["evidence_paths"]
        if "*" not in path and not (REPO_ROOT / path).exists()
    ]
    assert missing == []


def test_evidence_agrees_with_the_audit(evidence, audit_report):
    """The bundle is a rendering of the audit, so disagreement means one was hand-written."""
    by_key = {c["check"]: c["result"] for c in audit_report["checks"]}
    for row in evidence["requirements"]:
        assert row["result"] == by_key[row["backing_check"]]


def test_audit_passed(audit_report):
    failed = [c["check"] for c in audit_report["checks"] if c["result"] != "PASS"]
    assert failed == [], f"audit checks failed: {failed}"


# ---------------------------------------------------------------------------
# Independent re-derivation
# ---------------------------------------------------------------------------


def test_shard_content_hashes_recompute(artifacts, shard_dir):
    """The whole replay guarantee rests on shards being byte-identical to their manifests."""
    manifests = [read(p) for p in sorted((artifacts / "manifests").glob("*.manifest.json"))]
    assert manifests
    for manifest in manifests:
        tokens_path = shard_dir / manifest["tokens_file"]
        assert tagged("sha256", sha256_file(tokens_path)) == manifest["content_hash"]
        on_disk = tokens_path.stat().st_size // np.dtype(TOKEN_DTYPE).itemsize
        assert on_disk == manifest["token_count"]


def test_admission_verdicts_recompute(artifacts):
    manifests = [read(p) for p in sorted((artifacts / "manifests").glob("*.manifest.json"))]
    for manifest in manifests:
        verdict, _ = decide_admission(manifest)
        assert verdict == manifest["admission"], manifest["shard_id"]
        assert admission_score(manifest) == manifest["admission_score"]


def test_all_three_admission_verdicts_occur(artifacts):
    """A gate that only ever returns one answer has not been exercised."""
    index = read(artifacts / "manifests" / "shard_index.json")
    assert index["admitted"] > 0
    assert index["held_for_review"] > 0
    assert index["blocked_from_training"] > 0


def test_ledger_verifies(events):
    result = verify_append_only(events)
    assert result["ok"], result["issues"]
    assert result["offsets_contiguous"]


def test_rollback_actually_happened(events, artifacts):
    """The crash must have superseded steps, or the resume proof was trivial."""
    result = verify_append_only(events)
    run_a = result["per_branch"]["run-a"]
    assert run_a["superseded_by_rollback"] > 0
    assert run_a["recovery_epochs"] == [0, 1]


def test_no_blocked_document_reached_an_admitted_shard(artifacts, shard_dir):
    """The firewall's only claim that really matters."""
    firewall = read(artifacts / "ledgers" / "firewall_report.json")
    blocked = {r["doc_id"] for r in firewall["blocked_documents"]}
    assert blocked, "the firewall blocked nothing, so this proves nothing"

    for path in sorted((artifacts / "manifests").glob("*.manifest.json")):
        manifest = read(path)
        if manifest["admission"] != "admitted_to_registry":
            continue
        shard = load_shard(shard_dir, manifest["shard_id"])
        leaked = [s.doc_id for s in shard.doc_spans if s.doc_id in blocked]
        assert leaked == [], f"{manifest['shard_id']} contains blocked documents {leaked}"


def test_committed_batches_reference_admitted_shards_only(artifacts, events):
    admitted = {
        read(p)["shard_id"]
        for p in (artifacts / "manifests").glob("*.manifest.json")
        if read(p)["admission"] == "admitted_to_registry"
    }
    for branch in {e["run_branch_id"] for e in events}:
        for event in effective_batches(events, branch).values():
            unknown = set(event["shard_ids"]) - admitted
            assert unknown == set(), f"step {event['global_step']} used {unknown}"


def test_throughput_buckets_reconcile(artifacts):
    performance = read(artifacts / "performance.json")
    fate = performance["token_fate"]
    assert (
        fate["useful_loss_bearing"] + fate["opus_rejected"]
        + fate["padding_and_context_waste"] == fate["total_prepared"]
    )
    assert 0 < fate["shares"]["useful"] < 1
    assert sum(fate["shares"].values()) == pytest.approx(1.0, abs=1e-3)


def test_recovery_modes_diverge_as_expected(artifacts):
    """Widget 14's three-way result: only ledger mode reproduces the recorded stream."""
    fork = read(artifacts / "reports" / "fork_report.json")
    assert fork["modes"]["ledger"]["matches_original"] is True
    assert fork["modes"]["fork"]["matches_original"] is False
    assert fork["modes"]["random"]["matches_original"] is False
    assert fork["modes"]["fork"]["branch_id"] != fork["modes"]["ledger"]["branch_id"]
    # The dangerous one: a different data stream still calling itself run-a.
    assert fork["modes"]["random"]["branch_id"] == fork["modes"]["ledger"]["branch_id"]


def test_replay_matched_every_step(artifacts):
    replay = read(artifacts / "reports" / "replay_report.json")
    assert replay["all_matched"]
    assert replay["steps_compared"] > 0
    assert replay["mismatched_steps"] == []


def test_resume_matched_the_pre_crash_ledger(artifacts):
    resume = read(artifacts / "reports" / "resume_report.json")
    assert resume["all_matched"]
    assert resume["steps_in_window"] > 0
    assert resume["next_batch_after_resume"]["matched"]


def test_no_opus_firewall_hit_was_rescued_by_a_floor(artifacts):
    records = [
        json.loads(line)
        for line in (artifacts / "ledgers" / "opus_decisions.jsonl")
        .read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert records
    rescued = [
        r for r in records
        if r["decision"] == "protected" and r["rejection_reason"] == "eval_firewall_overlap"
    ]
    assert rescued == []


def test_learning_ledger_is_backed_by_measured_loss(artifacts):
    learning = read(artifacts / "ledgers" / "learning_ledger.json")
    assert learning["by_shard"]
    for row in learning["by_shard"].values():
        assert row["tokens_scored"] > 0
        assert row["mean_loss"] > 0
        assert row["v6_policy_hint"] in {
            "keep_with_phase_guard", "monitor_next_phase", "delay_or_reclean"
        }


def test_learning_ledger_stage_trend_is_in_curriculum_order(artifacts, config):
    """These files are written with sorted keys, so an ordered trend must be a list.

    A dict keyed by stage name would come back alphabetically -- anneal, foundation,
    skill_build -- and a reader would draw the curve backwards.
    """
    learning = read(artifacts / "ledgers" / "learning_ledger.json")
    order = [s["stage"] for s in config.require("mixture.stages")]
    for row in list(learning["by_shard"].values()) + list(learning["by_lane"].values()):
        trend = row["loss_trend_by_stage"]
        assert isinstance(trend, list)
        stages = [point["stage"] for point in trend]
        assert stages == [s for s in order if s in stages], stages
        if len(trend) >= 2:
            expected = round(trend[-1]["mean_loss"] - trend[0]["mean_loss"], 4)
            assert row["loss_delta"] == expected


def test_learning_ledger_reports_shard_coverage(artifacts):
    """'Never opened' and 'judged unhelpful' are different states."""
    learning = read(artifacts / "ledgers" / "learning_ledger.json")
    index = read(artifacts / "manifests" / "shard_index.json")
    coverage = learning["coverage"]
    assert coverage["admitted_shards"] == index["admitted"]
    assert coverage["shards_with_loss_attribution"] == len(learning["by_shard"])
    assert (coverage["shards_with_loss_attribution"] + coverage["shards_not_reached"]
            == coverage["admitted_shards"])


def test_all_four_opus_outcomes_occur(artifacts):
    """Accept, reject, defer and protect should each be exercised by the run itself."""
    records = [
        json.loads(line)
        for line in (artifacts / "ledgers" / "opus_decisions.jsonl")
        .read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    observed = {r["decision"] for r in records}
    assert observed == {"accepted", "rejected", "deferred", "protected"}, observed
