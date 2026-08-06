"""The eval firewall and the two-tier admission gate.

These are two independent gates and the tests keep them independent. The firewall decides
whether a *document* may enter the training stream at all; the admission gate decides
whether a *shard*, given its metadata, is fit to be scheduled and replayed. A document can
pass the firewall and still be stopped by the gate, and the run relies on both.
"""

from __future__ import annotations

import pytest

from tds.corpus import Document, Segment
from tds.firewall import EvalFirewall, ngram_hashes
from tds.manifest import (ADMITTED, BLOCKED, HELD, admission_score, decide_admission)


def make_doc(doc_id, text, registry=None, lane="web") -> Document:
    return Document(
        doc_id=doc_id, lane=lane, segments=[Segment("text", text)],
        source={"origin": "test"},
        registry=registry or {"never_train": False, "license_tier": "safe",
                              "dedup_status": "passed", "pii_screen_status": "screened"},
    )


@pytest.fixture(scope="module")
def firewall(config) -> EvalFirewall:
    return EvalFirewall(config)


# ---------------------------------------------------------------------------
# Firewall
# ---------------------------------------------------------------------------


def test_clean_document_is_admitted(firewall):
    verdict = firewall.scan(make_doc("clean", "A note about shard rotation policy in the "
                                              "pipeline, written for internal readers."))
    assert verdict.admitted
    assert verdict.clauses == []


def test_never_train_flag_blocks(firewall):
    doc = make_doc("held-out", "some held-out text",
                   registry={"never_train": True, "license_tier": "safe",
                             "dedup_status": "passed", "pii_screen_status": "screened"})
    verdict = firewall.scan(doc)
    assert not verdict.admitted
    assert "never_train flag" in verdict.clauses


def test_canary_string_blocks(firewall, config):
    prefix = config.require("eval_firewall.canary_prefix")
    verdict = firewall.scan(make_doc("canary", f"routine text {prefix}-abc-123 more text"))
    assert not verdict.admitted
    assert "canary match" in verdict.clauses


def test_verbatim_benchmark_text_blocks_on_overlap(firewall):
    example = firewall.benchmarks["GSM8K"][0]
    verdict = firewall.scan(make_doc("mirror", example))
    assert not verdict.admitted
    assert any(c.startswith("benchmark overlap") for c in verdict.clauses)


def test_trainable_flag_does_not_protect_a_derived_explanation(firewall):
    """Widget 13's central case: the registry says trainable and it is blocked anyway.

    A firewall that only reads a boolean would let this through, which is exactly how
    benchmark knowledge leaks in through blog posts and worked solutions.
    """
    example = firewall.benchmarks["GSM8K"][1]
    doc = make_doc(
        "derived",
        "Solving a classic word problem, step by step\n\n"
        f"{example}\n\nHere is how I would walk a student through it.",
    )
    assert doc.registry["never_train"] is False
    verdict = firewall.scan(doc)
    assert not verdict.admitted
    assert verdict.checks["never_train_flag"]["registry_flag"] == "trainable"
    assert any(c.startswith("benchmark overlap") for c in verdict.clauses)
    assert "benchmark-derived content" in verdict.clauses


def test_explanation_markers_alone_do_not_block(firewall):
    """A tutorial that shares no benchmark n-grams is just a tutorial."""
    verdict = firewall.scan(make_doc(
        "tutorial",
        "Here is how to rotate shards, step by step. First read the manifest, then "
        "compare the content hash, then append the new offset to the ledger.",
    ))
    assert verdict.admitted, verdict.reason


def test_each_check_contributes_its_own_clause(firewall):
    """Auditability: 'why was this blocked' must have a decomposable answer."""
    example = firewall.benchmarks["MMLU"][0]
    doc = make_doc("everything", example,
                   registry={"never_train": True, "license_tier": "safe",
                             "dedup_status": "passed", "pii_screen_status": "screened"})
    verdict = firewall.scan(doc)
    assert len(verdict.clauses) >= 2
    assert set(verdict.checks) == {
        "never_train_flag", "ngram_overlap", "canary_match", "derived_content"
    }


def test_short_documents_still_get_a_fingerprint():
    """A two-line benchmark item must not slip through by being shorter than the window."""
    assert ngram_hashes("only four words here", 13)
    assert ngram_hashes("", 13) == set()


# ---------------------------------------------------------------------------
# Admission gate
# ---------------------------------------------------------------------------


def complete_manifest(**overrides) -> dict:
    manifest = {
        "shard_id": "shard_web_0", "capability_lane": "web", "token_count": 100,
        "tokenizer_hash": "tok_abc", "content_hash": "sha256_abc",
        "cleaning_pipeline_hash": "clean_abc", "dedup_status": "passed",
        "pii_screen_status": "screened", "eval_overlap_status": "clear",
        "license_tier": "safe", "parent_manifest_ids": ["shard_upstream_1"],
    }
    manifest.update(overrides)
    return manifest


def test_complete_manifest_is_admitted():
    verdict, reasons = decide_admission(complete_manifest())
    assert verdict == ADMITTED
    assert admission_score(complete_manifest()) == 100


@pytest.mark.parametrize("field", ["pii_screen_status", "eval_overlap_status",
                                   "tokenizer_hash", "content_hash"])
def test_missing_hard_requirement_blocks(field):
    verdict, reasons = decide_admission(complete_manifest(**{field: None}))
    assert verdict == BLOCKED
    assert any(field in reason for reason in reasons)


@pytest.mark.parametrize("overrides", [
    {"dedup_status": None},
    {"license_tier": "unknown"},
    {"license_tier": "unsafe"},
])
def test_weak_soft_requirement_holds_for_review(overrides):
    """Held is not blocked: reproducibility is weakened, nothing unsafe is happening."""
    verdict, reasons = decide_admission(complete_manifest(**overrides))
    assert verdict == HELD
    assert reasons


def test_review_licence_alone_still_admits():
    """Widget 6: a 'review' licence with every other field present scored 89 and admitted."""
    verdict, _ = decide_admission(complete_manifest(license_tier="review"))
    assert verdict == ADMITTED


def test_missing_parent_manifests_admits_but_scores_lower():
    manifest = complete_manifest(parent_manifest_ids=[])
    verdict, reasons = decide_admission(manifest)
    assert verdict == ADMITTED
    assert any("lineage" in reason for reason in reasons)
    assert admission_score(manifest) < admission_score(complete_manifest())


def test_eval_overlap_blocked_marker_is_a_hard_block():
    verdict, _ = decide_admission(complete_manifest(eval_overlap_status="blocked_or_unknown"))
    assert verdict == BLOCKED
