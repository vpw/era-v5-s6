"""Documents entering the system, with provenance attached.

Five capability lanes. Two of them (`web`, `indic`) are real documents inherited from
S4's cleaning pipeline; three (`code`, `reasoning`, `agentic`) are generated here,
because S4's Sangraha corpus has no code, no proof traces and no tool trajectories.
Generated documents say so in their provenance -- nothing pretends to be more real than
it is.

Documents carry *segments*, not just text. A plain pretraining document is one segment;
an agentic trajectory is a sequence of user / assistant / tool_call / tool_observation
segments. That structure is what lets `masks.py` give the agentic lane a loss mask where
loss-bearing tokens are strictly fewer than non-pad tokens, matching widget 3.

This module also injects the contaminated candidates the eval firewall exists to catch:
verbatim benchmark mirrors, a canary-carrying document, and -- the case worth getting
right -- a benchmark-derived explanation whose own registry flag says `trainable`.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import REPO_ROOT, Config
from .hashing import sha256_text, short, stable_choice, stable_uniform

FIXTURES = REPO_ROOT / "fixtures"

# Segment roles. Only `assistant` and `tool_call` carry loss in the agentic lane.
ROLE_TEXT = "text"
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL_CALL = "tool_call"
ROLE_TOOL_OBSERVATION = "tool_observation"


@dataclass(frozen=True)
class Segment:
    role: str
    text: str


@dataclass
class Document:
    doc_id: str
    lane: str
    segments: list[Segment]
    source: dict
    registry: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.segments)

    @property
    def content_hash(self) -> str:
        return sha256_text(self.text)


def _doc_id(lane: str, seed_text: str) -> str:
    return f"{lane}-{short(sha256_text(seed_text), 10)}"


# ---------------------------------------------------------------------------
# Real documents, inherited from S4
# ---------------------------------------------------------------------------


def _load_s4_documents() -> list[dict]:
    path = FIXTURES / "upstream" / "s4_docs.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _s4_registry(record: dict) -> dict:
    """Safety and licence state inherited from S4's pipeline.

    S4 admitted every one of these documents: they are normalised, language-identified,
    quality-filtered, deduplicated, PII-screened and decontaminated. The one distinction
    that survives into S6 is the source pool -- S4's `verified` pool came from sources
    with a checked licence, the `unverified` pool did not, and that maps onto the licence
    tier the admission gate reads.
    """
    verified = record.get("pool") == "verified"
    return {
        "never_train": False,
        "license_tier": "safe" if verified else "review",
        "dedup_status": "passed",
        "pii_screen_status": "screened",
        "source_pool": record.get("pool"),
    }


def load_real_documents(config: Config) -> list[Document]:
    lang_to_lane = config.require("corpus.s4_lang_to_lane")
    wanted = config.require("corpus.docs_per_lane")

    by_lane: dict[str, list[Document]] = {}
    for record in _load_s4_documents():
        lang = record.get("detected_lang") or record.get("claimed_lang")
        lane = lang_to_lane.get(lang)
        if lane is None:
            continue
        doc = Document(
            doc_id=_doc_id(lane, record["upstream_doc_id"]),
            lane=lane,
            segments=[Segment(ROLE_TEXT, record["text"])],
            source={
                "origin": "s4",
                "upstream_doc_id": record["upstream_doc_id"],
                "upstream_shard_file": record["upstream_shard_file"],
                "source_url": "https://huggingface.co/datasets/ai4bharat/sangraha",
                "language": lang,
                "src": record["src"],
            },
            registry=_s4_registry(record),
        )
        by_lane.setdefault(lane, []).append(doc)

    out: list[Document] = []
    for lane in sorted(by_lane):
        docs = sorted(by_lane[lane], key=lambda d: d.doc_id)
        out.extend(docs[: wanted.get(lane, len(docs))])
    return out


# ---------------------------------------------------------------------------
# Generated documents: code, reasoning, agentic
# ---------------------------------------------------------------------------

_CODE_TASKS = [
    ("load_shard", "path", "list", "read one tokenized shard from disk"),
    ("pack_batch", "docs", "dict", "pack documents into a fixed window"),
    ("apply_mask", "tokens", "list", "zero out the non-loss-bearing positions"),
    ("compile_mixture", "plan", "dict", "turn lane weights into a token schedule"),
    ("verify_hash", "blob", "str", "recompute the content hash of a shard"),
    ("resume_run", "ledger", "int", "find the last committed ledger offset"),
    ("score_candidate", "batch", "float", "assign a proxy utility score"),
    ("split_spans", "seq", "list", "cut a packed sequence at EOS boundaries"),
]
_CODE_BODIES = [
    "    total = 0\n    for item in {arg}:\n        total += len(item)\n    return total\n",
    "    if not {arg}:\n        raise ValueError('empty input')\n    return sorted({arg})\n",
    "    result = []\n    for i, item in enumerate({arg}):\n        if i % 2 == 0:\n            result.append(item)\n    return result\n",
    "    seen = set()\n    for item in {arg}:\n        if item in seen:\n            continue\n        seen.add(item)\n    return len(seen)\n",
    "    with open({arg}, 'rb') as fh:\n        payload = fh.read()\n    return hashlib.sha256(payload).hexdigest()\n",
]


def _generate_code_documents(count: int, seed: int) -> list[Document]:
    docs = []
    for i in range(count):
        name, arg, ret, purpose = stable_choice(_CODE_TASKS, seed, "code-task", i)
        body = stable_choice(_CODE_BODIES, seed, "code-body", i).format(arg=arg)
        helper = stable_choice(["util", "core", "io", "sched"], seed, "code-mod", i)
        text = (
            f"# {helper}.py\n"
            "import hashlib\n\n\n"
            f"def {name}({arg}) -> {ret}:\n"
            f'    """{purpose[0].upper() + purpose[1:]}."""\n'
            f"{body}\n\n"
            f"def test_{name}():\n"
            f"    assert {name}({arg!r}) is not None\n"
        )
        docs.append(
            Document(
                doc_id=_doc_id("code", f"code-{seed}-{i}"),
                lane="code",
                segments=[Segment(ROLE_TEXT, text)],
                source={
                    "origin": "generated",
                    "generator": "tds.corpus._generate_code_documents",
                    "reason": "S4's corpus carries no source code; this lane is synthesised",
                    "language": "python",
                },
                registry={
                    "never_train": False,
                    "license_tier": "safe",
                    "dedup_status": "passed",
                    "pii_screen_status": "screened",
                },
            )
        )
    return docs


_REASON_SUBJECTS = [
    ("a shard holds {a} tokens and a window holds {b}", "how many full windows fit"),
    ("a lane needs {a} tokens and supply is {b}", "how many tokens are short"),
    ("{a} documents pack into windows of {b}", "how many sequences result"),
    ("a run commits {a} batches every {b} steps", "how many batches per 100 steps"),
]


def _generate_reasoning_documents(count: int, seed: int) -> list[Document]:
    docs = []
    for i in range(count):
        template, question = stable_choice(_REASON_SUBJECTS, seed, "reason-t", i)
        a = 120 + int(stable_uniform(seed, "reason-a", i) * 880)
        b = 8 + int(stable_uniform(seed, "reason-b", i) * 56)
        quotient, remainder = divmod(a, b)
        text = (
            f"Problem. Suppose {template.format(a=a, b=b)}. Determine {question}.\n"
            f"Step 1. Identify the two quantities: {a} and {b}.\n"
            f"Step 2. The question asks for a whole count, so divide {a} by {b}.\n"
            f"Step 3. {a} = {b} * {quotient} + {remainder}.\n"
            f"Step 4. The remainder {remainder} does not fill another whole unit.\n"
            f"Answer. {quotient}.\n"
        )
        docs.append(
            Document(
                doc_id=_doc_id("reasoning", f"reason-{seed}-{i}"),
                lane="reasoning",
                segments=[Segment(ROLE_TEXT, text)],
                source={
                    "origin": "generated",
                    "generator": "tds.corpus._generate_reasoning_documents",
                    "reason": "no reasoning traces in the inherited corpus",
                    "note": "arithmetic is generated and checked, not copied from a benchmark",
                },
                registry={
                    "never_train": False,
                    "license_tier": "safe",
                    "dedup_status": "passed",
                    "pii_screen_status": "screened",
                },
            )
        )
    return docs


_AGENT_GOALS = [
    ("find the token count of the indic shard", "shard_stats", "{'tokens': 128411, 'lane': 'indic'}"),
    ("check whether checkpoint 3 was bound to a ledger offset", "ledger_query", "{'offset': 148, 'ckpt': 'ckpt_00003'}"),
    ("look up the packing utilisation of the last run", "perf_query", "{'utilisation': 0.86}"),
    ("confirm the tokenizer hash on the code shard", "manifest_read", "{'tokenizer_hash': 'tok_4d4543e296a4'}"),
    ("list the lanes that fell below their protected floor", "floor_check", "{'below_floor': ['agentic']}"),
]


def _generate_agentic_documents(count: int, seed: int) -> list[Document]:
    """Multi-turn tool trajectories.

    Deliberately structured so that user turns and tool observations are context the
    model should condition on but never be scored against, while the assistant's own
    planning, tool calls and final answer carry loss. This is the lane where
    loss-bearing tokens < non-pad tokens.
    """
    docs = []
    for i in range(count):
        goal, tool, observation = stable_choice(_AGENT_GOALS, seed, "agent-goal", i)
        retry = stable_uniform(seed, "agent-retry", i) < 0.3
        segments = [
            Segment(ROLE_SYSTEM, "You are a data pipeline assistant with tool access.\n"),
            Segment(ROLE_USER, f"User: Please {goal}.\n"),
            Segment(ROLE_ASSISTANT, "Thought: I need to query the pipeline metadata store.\n"),
            Segment(ROLE_TOOL_CALL, f"Action: {tool}(scope='latest_run')\n"),
        ]
        if retry:
            segments += [
                Segment(ROLE_TOOL_OBSERVATION, "Observation: 404 not found\n"),
                Segment(ROLE_ASSISTANT, "Thought: the scope was wrong; retry against the cache.\n"),
                Segment(ROLE_TOOL_CALL, f"Action: {tool}(scope='cache')\n"),
            ]
        segments += [
            Segment(ROLE_TOOL_OBSERVATION, f"Observation: {observation}\n"),
            Segment(ROLE_ASSISTANT, f"Answer: {observation} -- read from the run manifest.\n"),
        ]
        docs.append(
            Document(
                doc_id=_doc_id("agentic", f"agent-{seed}-{i}"),
                lane="agentic",
                segments=segments,
                source={
                    "origin": "generated",
                    "generator": "tds.corpus._generate_agentic_documents",
                    "reason": "agentic trajectories are the scarcest lane; S5 flagged the shortfall",
                    "turns": len(segments),
                },
                registry={
                    "never_train": False,
                    "license_tier": "safe",
                    "dedup_status": "passed",
                    "pii_screen_status": "screened",
                },
            )
        )
    return docs


# ---------------------------------------------------------------------------
# Evaluation holdout and the contaminated candidates the firewall must catch
# ---------------------------------------------------------------------------


def load_eval_benchmarks() -> dict[str, list[str]]:
    """The held-out benchmark text. Never a training candidate; the firewall's reference."""
    with open(FIXTURES / "eval" / "benchmarks.json", encoding="utf-8") as fh:
        return json.load(fh)


def build_contaminated_candidates(config: Config) -> list[Document]:
    """Documents that *look* admissible and must not reach a loss-bearing position.

    Four kinds, mirroring widget 13's candidate stream:

      benchmark_mirror     verbatim eval text, honestly flagged never_train
      derived_explanation  a blog-style walkthrough quoting a benchmark question at
                           length, whose registry flag says `trainable` -- this is the
                           one that proves a flag check alone is not a firewall
      canary_carrier       carries a planted canary GUID
      near_duplicate       an eval item with light edits, flagged trainable
    """
    benchmarks = load_eval_benchmarks()
    canary_prefix = config.require("eval_firewall.canary_prefix")
    candidates: list[Document] = []

    def add(kind: str, lane: str, text: str, registry: dict, note: str) -> None:
        candidates.append(
            Document(
                doc_id=_doc_id(lane, f"contaminated-{kind}-{len(candidates)}"),
                lane=lane,
                segments=[Segment(ROLE_TEXT, text)],
                source={
                    "origin": "injected",
                    "contamination_kind": kind,
                    "reason": note,
                    "generator": "tds.corpus.build_contaminated_candidates",
                },
                registry=registry,
            )
        )

    gsm = benchmarks["GSM8K"]
    mmlu = benchmarks["MMLU"]
    milu = benchmarks["MILU-Hindi"]

    for i in range(2):
        add(
            "benchmark_mirror",
            "web",
            mmlu[i] + "\n\n" + mmlu[i + 2] + "\n",
            {"never_train": True, "license_tier": "review", "dedup_status": "passed",
             "pii_screen_status": "screened"},
            "a scraped mirror of a public benchmark; the registry flag is honest here",
        )

    for i in range(2):
        add(
            "derived_explanation",
            "web",
            "Solving a classic word problem, step by step\n\n"
            "A reader wrote in asking about this one:\n\n"
            f"{gsm[i]}\n\n"
            "Here is how I would walk a student through it, slowly.\n",
            # Flagged trainable on purpose: the registry believes this is a blog post.
            {"never_train": False, "license_tier": "safe", "dedup_status": "passed",
             "pii_screen_status": "screened"},
            "benchmark-derived explanation flagged trainable; must still be blocked",
        )

    add(
        "canary_carrier",
        "web",
        "Internal note on shard rotation policy. "
        f"{canary_prefix}-4f2a91c7-0b3d-4e88-9a51-7c6e2d40b915 "
        "Rotation happens whenever a manifest is superseded.\n",
        {"never_train": False, "license_tier": "safe", "dedup_status": "passed",
         "pii_screen_status": "screened"},
        "carries a planted canary GUID; presence proves a leak path",
    )

    add(
        "near_duplicate",
        "indic",
        milu[0].replace("  ", " ") + "\n" + milu[1] + "\n",
        {"never_train": False, "license_tier": "review", "dedup_status": "passed",
         "pii_screen_status": "screened"},
        "lightly edited Indic benchmark items; near-duplicate rather than exact",
    )

    # Two candidates that are clean as *content* but incomplete as *metadata*, so the
    # manifest gate -- not the firewall -- is what stops them. The gates are separate,
    # and they fail differently: a missing PII screen is a hard block, a missing dedup
    # status is only enough to hold the shard for review.
    add(
        "unscreened_but_clean",
        "code",
        "# scratch.py\n\ndef rotate(shards):\n    return shards[1:] + shards[:1]\n",
        {"never_train": False, "license_tier": "safe", "dedup_status": "passed",
         "pii_screen_status": None},
        "clean content, missing PII screen; hard-blocked by the manifest gate",
    )

    add(
        "undeduped_but_clean",
        "code",
        "# rotate_test.py\n\ndef test_rotate():\n    assert rotate([1, 2, 3]) == [2, 3, 1]\n",
        {"never_train": False, "license_tier": "unknown", "dedup_status": None,
         "pii_screen_status": "screened"},
        "clean content, no dedup status and an unknown licence; held for review, not blocked",
    )

    return candidates


# ---------------------------------------------------------------------------


def build_corpus(config: Config) -> list[Document]:
    """All candidate documents, in a deterministic order."""
    seed = config.require("run.seed")
    wanted = config.require("corpus.docs_per_lane")

    docs = load_real_documents(config)
    docs += _generate_code_documents(wanted["code"], seed)
    docs += _generate_reasoning_documents(wanted["reasoning"], seed)
    docs += _generate_agentic_documents(wanted["agentic"], seed)
    docs += build_contaminated_candidates(config)

    docs.sort(key=lambda d: (d.lane, d.doc_id))
    return docs


def corpus_summary(docs: list[Document]) -> dict:
    summary: dict[str, dict] = {}
    for doc in docs:
        entry = summary.setdefault(
            doc.lane, {"documents": 0, "characters": 0, "origins": {}}
        )
        entry["documents"] += 1
        entry["characters"] += len(doc.text)
        origin = doc.source.get("origin", "unknown")
        entry["origins"][origin] = entry["origins"].get(origin, 0) + 1
    return summary
