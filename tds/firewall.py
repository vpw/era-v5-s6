"""The evaluation and validation firewall.

Widget 13's teaching point is the whole design here: a shard can be flagged `trainable`
in the registry and still leak evaluation knowledge. So the firewall does not ask one
question. It runs four **independent** checks, and each one that fires contributes its
own clause to the rejection reason:

    never_train_flag   the registry says this is held-out data
    ngram_overlap      13-gram fingerprints match the held-out benchmark sets
    canary_match       a planted canary GUID is present
    derived_content    benchmark-derived explanation or walkthrough

A document is blocked if any clause fires. Recording the clauses rather than a boolean is
what makes the block auditable afterwards -- "why was this shard rejected" has an answer,
and the four reasons mean different things for V6 data planning.

The 13-gram window and the two-hit rule are inherited from S4's `decontaminate.py`, so
S6 scans with the same decontamination standard the corpus was cleaned under.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from .config import Config
from .corpus import Document, load_eval_benchmarks

WORD_RE = re.compile(r"\w+", re.UNICODE)

# Markers of a benchmark walkthrough: text that explains a test item rather than
# reproducing it. On its own this is innocent -- it only matters alongside overlap.
DERIVED_MARKERS = (
    "step by step",
    "here is how",
    "walk a student",
    "worked solution",
    "let me explain",
    "solving a classic",
    "answer key",
    "model answer",
)


def ngram_hashes(text: str, n: int) -> set[bytes]:
    """The set of n-gram fingerprints in `text`.

    Short documents still get one fingerprint over all their words, so a two-line
    benchmark item cannot slip through simply by being shorter than the window.
    """
    words = WORD_RE.findall(text.lower())
    if not words:
        return set()
    if len(words) < n:
        return {hashlib.blake2b(" ".join(words).encode("utf-8"), digest_size=8).digest()}
    return {
        hashlib.blake2b(" ".join(words[i : i + n]).encode("utf-8"), digest_size=8).digest()
        for i in range(len(words) - n + 1)
    }


@dataclass
class FirewallVerdict:
    doc_id: str
    lane: str
    admitted: bool
    clauses: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)

    @property
    def reason(self) -> str:
        if self.admitted:
            return "registered access and admitted to train stream"
        return "rejected: " + ", ".join(self.clauses)

    def to_record(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "lane": self.lane,
            "verdict": "admitted" if self.admitted else "blocked",
            "clauses": list(self.clauses),
            "reason": self.reason,
            "checks": self.checks,
        }


class EvalFirewall:
    def __init__(self, config: Config):
        fw = config.require("eval_firewall")
        self.n = fw["ngram_n"]
        self.min_hits = fw["min_ngram_hits"]
        self.blocking_overlap_pct = fw["blocking_overlap_pct"]
        self.canary_prefix = fw["canary_prefix"]
        self.enabled = dict(fw["checks_enabled"])

        self.benchmarks = load_eval_benchmarks()
        self.fingerprints: set[bytes] = set()
        self.per_set_counts: dict[str, int] = {}
        for name in sorted(self.benchmarks):
            hashes: set[bytes] = set()
            for example in self.benchmarks[name]:
                hashes |= ngram_hashes(example, self.n)
            self.per_set_counts[name] = len(hashes)
            self.fingerprints |= hashes

    # -- the four checks ---------------------------------------------------------

    def _check_never_train(self, doc: Document) -> tuple[bool, dict]:
        flagged = bool(doc.registry.get("never_train"))
        return flagged, {"registry_flag": "never_train" if flagged else "trainable"}

    def _check_overlap(self, doc: Document) -> tuple[bool, dict]:
        doc_hashes = ngram_hashes(doc.text, self.n)
        if not doc_hashes:
            return False, {"overlap_pct": 0.0, "ngram_hits": 0, "ngrams_total": 0}
        hits = len(doc_hashes & self.fingerprints)
        pct = round(100.0 * hits / len(doc_hashes), 2)
        fires = hits >= self.min_hits or pct >= self.blocking_overlap_pct
        return fires, {
            "overlap_pct": pct,
            "ngram_hits": hits,
            "ngrams_total": len(doc_hashes),
        }

    def _check_canary(self, doc: Document) -> tuple[bool, dict]:
        present = self.canary_prefix in doc.text
        return present, {"canary_scan": "match" if present else "clear"}

    def _check_derived(self, doc: Document, overlap: dict) -> tuple[bool, dict]:
        """Explanation markers only count as derived content alongside real overlap.

        A tutorial that shares no benchmark n-grams is just a tutorial.
        """
        lowered = doc.text.lower()
        marker = next((m for m in DERIVED_MARKERS if m in lowered), None)
        fires = marker is not None and overlap.get("ngram_hits", 0) >= 1
        return fires, {
            "derived_data": "benchmark derivative" if fires else "clear",
            "marker": marker,
        }

    # -- driver ------------------------------------------------------------------

    def scan(self, doc: Document) -> FirewallVerdict:
        clauses: list[str] = []
        checks: dict = {}

        fires, detail = self._check_never_train(doc)
        checks["never_train_flag"] = detail
        if fires and self.enabled.get("never_train_flag", True):
            clauses.append("never_train flag")

        fires, overlap = self._check_overlap(doc)
        checks["ngram_overlap"] = overlap
        if fires and self.enabled.get("ngram_overlap", True):
            clauses.append(f"benchmark overlap {overlap['overlap_pct']}%")

        fires, detail = self._check_canary(doc)
        checks["canary_match"] = detail
        if fires and self.enabled.get("canary_match", True):
            clauses.append("canary match")

        fires, detail = self._check_derived(doc, overlap)
        checks["derived_content"] = detail
        if fires and self.enabled.get("derived_content", True):
            clauses.append("benchmark-derived content")

        return FirewallVerdict(
            doc_id=doc.doc_id,
            lane=doc.lane,
            admitted=not clauses,
            clauses=clauses,
            checks=checks,
        )

    def scan_all(self, docs: list[Document]) -> tuple[list[Document], list[FirewallVerdict]]:
        """Split a corpus into what may train and the full record of what was refused."""
        admitted: list[Document] = []
        verdicts: list[FirewallVerdict] = []
        for doc in docs:
            verdict = self.scan(doc)
            verdicts.append(verdict)
            if verdict.admitted:
                admitted.append(doc)
        return admitted, verdicts

    def descriptor(self) -> dict:
        return {
            "ngram_n": self.n,
            "min_ngram_hits": self.min_hits,
            "blocking_overlap_pct": self.blocking_overlap_pct,
            "canary_prefix": self.canary_prefix,
            "checks_enabled": dict(self.enabled),
            "eval_sets": {
                name: {
                    "examples": len(self.benchmarks[name]),
                    "distinct_ngrams": self.per_set_counts[name],
                }
                for name in sorted(self.benchmarks)
            },
            "fingerprints_total": len(self.fingerprints),
        }


def firewall_report(verdicts: list[FirewallVerdict], firewall: EvalFirewall) -> dict:
    """The artifact written to ledgers/firewall_report.json."""
    blocked = [v for v in verdicts if not v.admitted]
    clause_counts: dict[str, int] = {}
    for verdict in blocked:
        for clause in verdict.clauses:
            key = clause.split(" ")[0] if clause.startswith("benchmark overlap") else clause
            key = "benchmark overlap" if clause.startswith("benchmark overlap") else clause
            clause_counts[key] = clause_counts.get(key, 0) + 1

    # The case worth calling out by name: blocked despite a `trainable` registry flag.
    blocked_despite_trainable = [
        v.doc_id
        for v in blocked
        if v.checks["never_train_flag"]["registry_flag"] == "trainable"
    ]

    return {
        "config": firewall.descriptor(),
        "scanned": len(verdicts),
        "admitted": len(verdicts) - len(blocked),
        "blocked": len(blocked),
        "clause_counts": clause_counts,
        "blocked_despite_trainable_flag": blocked_despite_trainable,
        "blocked_documents": [v.to_record() for v in blocked],
    }
