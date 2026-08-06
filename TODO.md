# S6 TODO — Building the Training Dataset (Training Data Execution System)

`S6-assignment.md` = the task, `PLAN.md` = architecture and build order,
`resources/s6-session.md` = lesson, `resources/s6-transcript.md` = live-class transcript,
`resources/s6-widget-data.md` = extracted widget content (real JSON schemas for the
ledger/OPUS/manifest/recovery widgets — implement from these, not from scratch).

## Done — scaffolding

- [x] Read `S6-assignment.md`.
- [x] Extract real widget data from the session page — all 15 widgets in
      `resources/s6-widget-data.md`. Widgets are standalone pages at
      `/widgets/s6_widget_N_*.html`.
- [x] Pull the full session prose into `resources/s6-session.md` (17 sections) and the
      live-class transcript into `resources/s6-transcript.md` (via the linked Google Doc's
      `/export?format=txt`).
- [x] Set up `CLAUDE.md` for this directory.
- [x] Decide model / corpus / repo strategy and write `PLAN.md`.
      NumPy-only tiny transformer; S4 real docs for web+indic with generated code/reasoning/
      agentic; S2 tokenizer; S5 `ledger.json` for weights and floors; standalone git repo.

## Phase 0 — foundation

- [ ] `git init` a standalone repo rooted at this directory. Decided: `work/` is ignored,
      the run of record's `submission_artifacts/` **is** committed (note it in README).
- [x] `.gitignore`, `requirements.txt` — numpy, tokenizers, pytest. No network at demo time.
- [x] `tds/hashing.py` — canonical JSON (sorted keys, fixed separators) + sha256 helpers,
      plus `stable_uniform` for reproducible pseudo-random quantities.
- [x] Determinism policy: hash-derived quantities keyed by explicit inputs; no reliance on
      dict/set iteration order or salted `hash()`; all hashes over bytes.
- [x] `config/run_config.json` — one file drives the whole demo; its SHA-256 travels into
      the evidence bundle. `tds/config.py` loads it and resolves paths.
- [x] `scripts/build_fixtures.py` + committed `fixtures/` (11.5 MB) — the S2/S4/S5 slices,
      so the repo clones and runs standalone with no network.

## Phase 1 — data into admitted shards

- [x] `tds/corpus.py` — 3,527 candidate documents. Web + Indic are real S4 decontaminated
      documents carrying upstream ids and pool/licence state; code, reasoning and agentic are
      generated (S4 has none) and say so in their provenance. Agentic documents carry
      role-tagged segments so the loss mask can differ by role. Injected edge cases: benchmark
      mirrors, two derived explanations flagged `trainable`, a canary carrier, a near-duplicate,
      and two metadata-incomplete documents that the *manifest gate* must stop.
- [x] `tds/tokenizer.py` — frozen sarvam1 vocab (68,096); `tokenizer_hash` = `tok_bb5115a36ddb`
      over the file bytes; special-token policy and `normalizer_id` recorded; encoding preserves
      segment roles. No path here trains or extends a vocabulary.
- [x] `tds/shards.py` — 33 immutable shards, 1.9M tokens. `<id>.tokens.bin` (uint32) +
      `<id>.docs.jsonl` (token spans + role spans + provenance). Writer refuses to overwrite an
      existing `shard_id`. Shards grouped by (lane, metadata signature) so one under-documented
      document quarantines itself instead of taking a whole shard down.
- [x] `tds/manifest.py` — widget 6 schema exactly, plus the **two-tier admission gate**.
      Verified on the real corpus: 31 admitted / 1 held (missing `dedup_status`, unknown
      licence) / 1 blocked (missing `pii_screen_status`). All three verdicts occur naturally.
- [ ] Assert widget 1's 15-field checklist has a home — shard-level in the manifest, batch-level
      in the ledger event. (Deferred to `audit.py`, which is where the assertion belongs.)
- [x] `tds/firewall.py` — 4 **independent** checks, 13-gram window and 2-hit rule inherited from
      S4's `decontaminate.py`, scanning against real MMLU/GSM8K/MILU-Hindi text. Result: 6 of
      3,527 blocked, **4 of them despite a `trainable` registry flag** — including the two
      derived explanations, reproducing widget 13's `gsm8k-rationale-blog` case. Zero false
      positives on S4's already-decontaminated documents.

## Phase 2 — schedule and selection

- [x] `tds/mixture.py` — reads S5's ledger for floors, epoch ceiling and the OPUS keep
      fraction (0.4, i.e. a 60% rejection rate); compiles widget 7's 3 stages into an explicit
      **per-microbatch lane plan** (1,200 slots) written out before training, so planned shares
      are exact and the audit can compare consumed shares against an artifact. Slots are spread
      by stride scheduling rather than arriving in blocks. Supply check uses three verdicts, not
      two: `satisfied_by_unique_tokens` / `covered_by_repetition` (within S5's epoch ceiling of
      4) / `shortfall`. Agentic needs 3.3 epochs — short on unique tokens under every profile,
      exactly widget 7's finding, and left that way on purpose.
- [ ] `tds/opus.py` — accept / reject / defer / protected with widget 10's exact decision-record
      schema and 5-value rejection taxonomy (+ `deferred_for_anneal`, `protected_floor_override`).
      **Proxy score must be deterministic** (hash of candidate content x lane/stage/model-age
      priors), never an RNG that advances with training — otherwise replay breaks.
- [ ] Encode the invariant: protected floors override score/quota rejections but **never** an
      `eval_firewall_overlap` hit.
- [ ] `tds/sampler.py` — `plan(branch_id, step) -> BatchSpec`, a pure function of seed, branch,
      step, schedule and admitted registry. This is the spine of resume/replay/fork.

## Phase 3 — batches

- [ ] `tds/packing.py` — all 5 policies (pad-each-doc, concat-and-chop, greedy, best-fit,
      structure-preserving) with utilization / unused positions / boundary risk, reported in
      widget 5's table shape. Per-lane defaults: web+indic concat-and-chop, code best-fit,
      reasoning+agentic structure-preserving.
- [ ] `tds/masks.py` — loss mask (agentic masks user turns + tool observations; plain lanes fully
      loss-bearing on non-pad), attention blocked across segments, position ids reset at EOS.
      Record `position_policy: "packed_reset_on_eos"` in the ledger.
- [ ] `batch_hash = sha256(input_ids, loss_mask, position_ids, segment_ids)` — batch identity is
      independent of model state. This is what replay compares.

## Phase 4 — training and ledger

- [ ] `tds/model.py` — tiny NumPy transformer, forward + hand-written backward, per-token loss.
      Validate with a finite-difference gradient check.
- [ ] `tds/trainer.py` — train loop; `global_step` <-> `ledger_offset`; measures its own timing
      for the throughput report.
- [ ] `tds/ledger.py` — append-only JSONL with widget 8's `batch_committed` /`checkpoint_bound` /
      `worker_crash_recovered` schemas verbatim (+ `batch_hash`, `packing_utilization`,
      `loss_bearing_tokens`). Token span ids in `<lane>_<step>_<mb>:<start>-<end>` form.
- [ ] `tds/checkpoint.py` — model + optimizer + `dataloader_state: "ledger_offset_<k>"` + rng
      state, written as `.npz` with an index.

## Phase 5 — crash, resume, replay, fork

- [ ] Run training phases as **subprocesses** so the crash is a hard `os._exit(1)` that destroys
      in-memory state — otherwise "expected == actual" on resume is tautological.
- [ ] Resume: fresh process reconstructs from checkpoint + ledger only; asserts the next batch
      hash matches `plan(branch, last_offset+1)`; separately scans the full ledger for contiguity
      and duplicate offsets. -> `resume_report.json`
- [ ] Replay: re-derive an earlier interval from the immutable shards and compare batch ids,
      token span ids, loss-mask hashes and batch hashes field-by-field. -> `replay_report.json`
- [ ] Fork: restore an earlier checkpoint as `run-b`, recording parent branch + parent step;
      `branch_id` changes **only** on fork. -> `fork_report.json`
- [ ] `random` mode as the **negative control** — same checkpoint, no ledger, reseeded sampler ->
      a different stream. Proves the ledger is what makes replay work. Three-way comparison table
      in `evidence.md` (widget 14).
- [ ] Use widget 14's run-level vocabulary: `run_started`, `trainer_crashed`,
      `checkpoint_restored{mode: ledger|fork|random}`.

## Phase 6 — learning ledger and performance

- [ ] `tds/learning.py` — per-token loss -> aggregate by (shard, lane, stage) -> first-vs-last
      exposure loss delta -> usefulness score (0-100) -> `v6_policy_hint`
      (`keep_with_phase_guard` / `delay_or_reclean`), in widget 12's report-card shape. Explicit
      joinable event chain: `batch_committed` -> `token_ppl_aggregated` ->
      `learning_delta_attached` -> `v6_policy_hint`.
- [ ] Check widget 11's prediction against our own trace (Indic hardest, and still hardest late)
      — report what we measure either way.
- [ ] `tds/perf.py` — widget 15's 4-way token fate, all **measured**: useful / opus_rejected /
      padding_waste / loader_wait. Headline: useful loss-bearing tokens/sec + packing
      utilization. -> `performance.json`, fully reconstructible from the ledger.

## Phase 7 — audit and evidence

- [ ] `tds/audit.py` — standalone entry point reading **only** `submission_artifacts/` + shard
      files. Re-derives: content/tokenizer hashes, every admission verdict, ledger contiguity,
      **batch hashes recomputed from shard bytes + recorded spans**, planned-vs-actual mixture
      shares and floor compliance, firewall leak scan, throughput arithmetic.
      -> `audit_report.json`
- [ ] `tds/evidence.py` — `evidence.json` + `evidence.md` generated **from the audit's findings**
      (the code path that writes a PASS is the path that verified it). `evidence.md` renders the
      assignment's exact 9 rows.
- [ ] `run_demo.py` — one command, no manual steps, produces the full tree:
      `run.log`, `evidence.json`, `evidence.md`, `performance.json`, `manifests/`, `ledgers/`,
      `checkpoints/`, `reports/`.
- [ ] `run.log` — all 13 required lifecycle events in order, with `[PASS]` markers for
      `tokenizer_hash_verified`, `eval_shard_blocked`, `checkpoint_saved`,
      `resume_next_batch_matched`, `replay_hash_matched`.

## Phase 8 — tests, docs, submission

- [ ] Invariant tests (pytest) — the 15 rows in `PLAN.md` §6: gradient check, tokenizer freeze,
      shard immutability, admission tiers, firewall (incl. no eval token at a loss-bearing
      position over the real run), packing/masks/positions, OPUS determinism + floor-vs-firewall,
      mixture floors + shortfall, ledger append-only/contiguous, resume, replay, fork,
      evidence-claims-resolve.
- [ ] Short README — architecture + design decisions, dense, no padding (same grading bias as
      prior sessions). Cross-reference which S1-S5 artifact satisfies which contract.
- [ ] Verify a clean-clone run: fresh venv, `pip install -r requirements.txt`,
      `python run_demo.py`, under 3 minutes, no network.
- [ ] Graded run of record — capture `submission_artifacts/` from that run.
- [ ] Push to GitHub, submit the link.

## After submission

- [ ] Merge the already-tested remote branches `s4-data-cleaning` and `s5-mixture-curriculum`
      into `master` on `github.com/vpw/TSAI`, then land S6 as an `s6-*` branch off master.
