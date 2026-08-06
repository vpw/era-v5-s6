"""Copy the slices of Sessions 2/4/5 that Session 6 consumes into `fixtures/`.

Run once, from a checkout that sits next to its sibling session directories; the output
is committed so the S6 repo clones and runs standalone with no network and no access to
the rest of the course tree.

    python3 scripts/build_fixtures.py

What comes from where, and why:

  S4  data/run/shards/*.jsonl        cleaned, language-identified, quality-filtered,
                                     deduped, PII-screened and decontaminated documents.
                                     These are the web and indic lanes. Using them is what
                                     makes `cleaning_pipeline_hash` and `eval_overlap_status`
                                     inherited facts rather than decoration.
  S4  data/run/manifests.json        the upstream shard manifests, including the SHA-256 of
                                     every cleaning script that touched the corpus. S6's
                                     manifests cite these as `parent_manifest_ids`.
  S4  data/raw/eval/*.json           the held-out benchmark sets S4 decontaminated against.
                                     S6 rebuilds 13-gram fingerprints from them so the eval
                                     firewall scans real benchmark text.
  S4  models/tokenizer-sarvam1.json  the frozen 68,096-entry vocab the S5 proxy trained on.
  S5  data/ledger.json               lane weights, protected floors, anneal reserve and the
                                     OPUS keep-fraction -- the mixture recipe S6 executes.

Sampling is deterministic (documents are ranked by the SHA-256 of their id), so re-running
this script reproduces the same fixtures byte for byte.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V5_ROOT = REPO_ROOT.parent.parent          # .../ERA/V5
S4 = V5_ROOT / "S4" / "assignment"
S5 = V5_ROOT / "S5" / "assignment"

FIXTURES = REPO_ROOT / "fixtures"
UPSTREAM = FIXTURES / "upstream"
EVAL = FIXTURES / "eval"
TOKENIZER = FIXTURES / "tokenizer"

# How many documents to carry over per source language. The web lane is capped by what
# S4 actually has (2,309 English documents); the Indic lanes are sampled well below
# their ceiling to keep the fixture small.
SAMPLE_PER_LANG = {"eng": 900, "hin": 900, "tel": 500}

# Benchmark examples per eval set. Enough real text for genuine 13-gram overlap without
# committing megabytes.
EVAL_SAMPLE = 300
EVAL_SETS = ["GSM8K", "MMLU", "MILU-Hindi"]


def rank_key(value: str) -> str:
    """Deterministic, well-spread ordering key -- avoids taking the first N of a file."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def require(path: Path, what: str) -> Path:
    if not path.exists():
        sys.exit(
            f"missing {what}: {path}\n"
            "build_fixtures.py must run from a checkout beside its sibling session "
            "directories (ERA/V5/S4, ERA/V5/S5). The committed fixtures/ tree already "
            "contains its output, so a fresh clone does not need to run this."
        )
    return path


def sample_s4_documents() -> list[dict]:
    shard_dir = require(S4 / "data" / "run" / "shards", "S4 cleaned shards")
    by_lang: dict[str, list[dict]] = {lang: [] for lang in SAMPLE_PER_LANG}

    for shard_path in sorted(shard_dir.glob("shard-*.jsonl")):
        with open(shard_path, encoding="utf-8") as fh:
            for line in fh:
                doc = json.loads(line)
                lang = doc.get("detected_lang") or doc.get("claimed_lang")
                if lang not in by_lang:
                    continue
                by_lang[lang].append(
                    {
                        "upstream_doc_id": doc["id"],
                        "upstream_shard_file": shard_path.name,
                        "src": doc["src"],
                        "pool": doc["pool"],
                        "claimed_lang": doc.get("claimed_lang"),
                        "detected_lang": doc.get("detected_lang"),
                        "text": doc["text"],
                    }
                )

    selected: list[dict] = []
    for lang, want in sorted(SAMPLE_PER_LANG.items()):
        pool = sorted(by_lang[lang], key=lambda d: rank_key(d["upstream_doc_id"]))
        if len(pool) < want:
            print(f"  note: only {len(pool)} {lang} documents available, wanted {want}")
        selected.extend(pool[:want])
        print(f"  {lang}: {min(want, len(pool))} of {len(pool)} documents")

    selected.sort(key=lambda d: d["upstream_doc_id"])
    return selected


def sample_eval_sets() -> dict:
    eval_dir = require(S4 / "data" / "raw" / "eval", "S4 cached eval sets")
    out: dict[str, list[str]] = {}
    for name in EVAL_SETS:
        with open(eval_dir / f"{name}.json", encoding="utf-8") as fh:
            examples = json.load(fh)
        examples = [e for e in examples if isinstance(e, str) and e.strip()]
        examples.sort(key=rank_key)
        out[name] = examples[:EVAL_SAMPLE]
        print(f"  {name}: {len(out[name])} of {len(examples)} examples")
    return out


def main() -> None:
    for directory in (UPSTREAM, EVAL, TOKENIZER):
        directory.mkdir(parents=True, exist_ok=True)

    provenance: dict = {
        "_doc": "Where each committed fixture came from. Regenerate with scripts/build_fixtures.py.",
        "sources": {},
    }

    print("S4 documents ->  fixtures/upstream/s4_docs.jsonl.gz")
    docs = sample_s4_documents()
    docs_path = UPSTREAM / "s4_docs.jsonl.gz"
    # mtime=0 keeps the gzip container byte-identical between runs.
    with gzip.GzipFile(docs_path, "wb", mtime=0) as gz:
        for doc in docs:
            gz.write((json.dumps(doc, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    provenance["sources"]["s4_docs"] = {
        "from": "S4/assignment/data/run/shards/*.jsonl (stage 06-decontaminated output)",
        "documents": len(docs),
        "per_language": SAMPLE_PER_LANG,
        "selection": "deterministic: ranked by sha256(upstream_doc_id), lowest N per language",
    }

    print("S4 manifests ->  fixtures/upstream/s4_manifests.json")
    src = require(S4 / "data" / "run" / "manifests.json", "S4 manifests")
    shutil.copyfile(src, UPSTREAM / "s4_manifests.json")
    provenance["sources"]["s4_manifests"] = {
        "from": "S4/assignment/data/run/manifests.json",
        "used_for": "parent_manifest_ids, cleaning_pipeline_hash, source_manifest lineage",
    }

    print("S5 ledger    ->  fixtures/upstream/s5_ledger.json")
    src = require(S5 / "data" / "ledger.json", "S5 mixture ledger")
    shutil.copyfile(src, UPSTREAM / "s5_ledger.json")
    provenance["sources"]["s5_ledger"] = {
        "from": "S5/assignment/data/ledger.json",
        "used_for": "lane weights, protected floors, anneal reserve, OPUS keep fraction",
    }

    print("tokenizer    ->  fixtures/tokenizer/tokenizer-sarvam1.json")
    src = require(S4 / "models" / "tokenizer-sarvam1.json", "sarvam1 tokenizer")
    shutil.copyfile(src, TOKENIZER / "tokenizer-sarvam1.json")
    provenance["sources"]["tokenizer"] = {
        "from": "S4/assignment/models/tokenizer-sarvam1.json",
        "note": "68,096-entry BPE vocab; the same tokenizer the S5 proxy ablation trained on",
    }

    print("eval sets    ->  fixtures/eval/benchmarks.json")
    benchmarks = sample_eval_sets()
    with open(EVAL / "benchmarks.json", "w", encoding="utf-8") as fh:
        json.dump(benchmarks, fh, ensure_ascii=False, indent=1, sort_keys=True)
    provenance["sources"]["eval_benchmarks"] = {
        "from": "S4/assignment/data/raw/eval/{GSM8K,MMLU,MILU-Hindi}.json",
        "examples_per_set": EVAL_SAMPLE,
        "used_for": "13-gram fingerprints for the eval firewall's overlap check",
    }

    # Hash every fixture so the audit can prove the committed inputs were not swapped.
    digests = {}
    for path in sorted(FIXTURES.rglob("*")):
        if path.is_file() and path.name != "PROVENANCE.json":
            digests[str(path.relative_to(FIXTURES))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    provenance["fixture_sha256"] = digests

    with open(FIXTURES / "PROVENANCE.json", "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, ensure_ascii=False, indent=1, sort_keys=True)

    total = sum(p.stat().st_size for p in FIXTURES.rglob("*") if p.is_file())
    print(f"\nfixtures/ is {total / 1e6:.1f} MB across {len(digests)} files")


if __name__ == "__main__":
    main()
