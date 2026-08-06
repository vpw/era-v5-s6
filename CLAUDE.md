# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

Session 6 (S6) assignment of the ERA V5 course (The School of AI). The session topic is
**Building the Training Dataset** — specifically the "Training Data Execution System" that
turns Session 5's mixture-and-curriculum *plan* into an actual, auditable data stream a
training loop consumes.

Unlike S5 (a written README plan) or S2-S4 (a Netlify widget/report), **this deliverable is a
working codebase**. Per `S6-assignment.md`, submit a GitHub repo containing:

- The complete implementation of the full path: `documents -> tokenized shards -> manifests ->
  mixture schedule -> packing -> batches -> training -> consumption ledger -> learning ledger
  -> checkpoint -> crash -> resume -> replay -> audit`.
- A short README explaining architecture and design decisions.
- One command (`python run_demo.py` or similar) that runs the complete demonstration with no
  manual intervention.
- Automated tests for the important invariants.
- A generated `submission_artifacts/` tree: `run.log`, `evidence.json`, `evidence.md`,
  `manifests/`, `ledgers/`, `checkpoints/`, `performance.json`.

The corpus, tokenizer, and model may all be small/toy — **the goal is not scale, it's proving
the data system is correct, reproducible, auditable, and efficient.** The final run must
deliberately crash, resume, and prove the next batch exactly matches what's expected; it must
also replay an earlier interval and prove the reconstructed batch ids/token spans/hashes match
the original run.

**Grading is evidence-first, not prose-first**: evaluators execute the one command, verify
`evidence.json`/`evidence.md` against the generated manifests/ledgers/reports, then inspect code
to confirm the evidence wasn't hardcoded or simulated. 1,000 points total, split roughly across
end-to-end execution (150), shard/manifest/tokenizer integrity (100), packing/masks/batch
correctness (150), mixture/floors/OPUS (150), ledgers (150), checkpoint/crash/resume/replay/fork
(150), eval firewall (50), throughput (50), tests/evidence/docs (50) — see `S6-assignment.md`
for the exact table.

## Layout

- `S6-assignment.md` — the assignment statement and evaluation rubric, verbatim.
- `TODO.md` — working checklist for this session; keep it updated as steps complete.
- `resources/s6-session.md` — full lesson writeup (17 sections), including the batch/padding/
  packing vocabulary, the exact stage-record and shard-manifest JSON shapes shown in the
  lesson, and descriptions of all 15 interactive widgets.
- `resources/s6-transcript.md` — full live-class transcript (~145KB, pulled from the session's
  linked Google Doc); mine it for implementation details and reasoning not in the session
  summary.
- `resources/s6-widget-data.md` — extracted live content from all 15 session widgets (real
  JSON event schemas for the ledger/OPUS/manifest widgets, real packing-policy utilization
  numbers, the exact crash/replay/fork state-machine behavior). **This is the most
  load-bearing resource file for implementation** — several widgets (8, 10, 14 especially)
  hand you the literal event/decision JSON schema to replicate in code, not just a prose
  description.
- Not yet created: the actual implementation (shard builder, manifest system, mixture
  compiler, packer, ledgers, checkpoint/crash/resume/replay/fork harness, tests, `run_demo.py`,
  and the generated `submission_artifacts/`).

## Conventions

- Submission target is a GitHub repo with runnable code — no Netlify deploy, no README-only
  submission like S5.
- This session is explicitly cumulative: the assignment text ties every subsystem back to a
  contract from Sessions 1-5 (next-token loss maps -> S1, tokenizer hashes -> S2, source
  provenance -> S3, cleaning/admission manifests -> S4, mixture/curriculum/protected-floors/
  OPUS -> S5). Widget 1 (`resources/s6-widget-data.md`) lays out this contract mapping
  explicitly — use it as the canonical checklist for what the shard manifest and ledger schemas
  must carry.
- Reuse real numbers/schemas from `resources/s6-widget-data.md` rather than inventing plausible
  ones — e.g. widget 8's `batch_committed`/`checkpoint_bound`/`worker_crash_recovered` JSON
  shapes, widget 10's OPUS decision-record shape and 5-reason rejection taxonomy, widget 14's
  exact recovery-mode vocabulary (`ledger` / `fork` / `random`), widget 6's shard-manifest JSON
  shape and two-tier admission gate (hard-block vs. held-for-review).
- Same "short and dense, no padding" grading bias as prior sessions carries over to the README,
  even though the bulk of the grade here is code + evidence rather than prose.
- Before adding new widget-derived numbers, check `resources/s6-widget-data.md` first — it was
  built via the `extract-widget-data` skill against the live session page and already covers
  all 15 widgets.
