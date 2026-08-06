# S6 implementation plan — Training Data Execution System

Architecture and build order. Feeds `README.md` at the end; `TODO.md` is the checklist view.

Source of truth for schemas and vocabulary: `resources/s6-widget-data.md`. Where a widget hands
over a literal JSON shape (6 manifest, 8 ledger events, 10 OPUS records, 14 recovery modes), we
mirror it rather than invent a parallel one.

---

## 0. Decisions taken

| Decision | Choice | Why |
|---|---|---|
| Model | **NumPy-only tiny transformer**, hand-written forward+backward | No torch anywhere in the course venvs. Grader installs `numpy` + `tokenizers` and it runs. Bit-deterministic on CPU. Correctness guarded by a finite-difference gradient check that doubles as a test. |
| Corpus | **S4 real docs for web/indic + locally generated code/reasoning/agentic** | S4's `06-decontaminated` output is genuinely cleaned, deduped, PII-screened and contamination-scanned, and carries real per-script SHA-256 cleaning hashes. Makes `cleaning_pipeline_hash` / `eval_overlap_status` inherited facts, not decoration. S4 has no code/reasoning/agentic lanes, so those are generated. |
| Tokenizer | **S2's trained `tokenizer.json`**, frozen and hashed | Satisfies the S2 contract (frozen vocab + merges, special-token policy, Indic normalization) with a real artifact. |
| Mixture inputs | **S5's `data/ledger.json`** (lane weights, protected floors, anneal reserve) | The S5 contract: the recipe becomes an executable schedule. |
| Repo | **Standalone git repo rooted at `S6/assignment/`** | Grader clones one URL and runs one command. Post-submission: merge the tested `s4-data-cleaning` and `s5-mixture-curriculum` remote branches into `master`, then land S6 as an `s6-*` branch. |
| Runtime target | **< 3 minutes, CPU, no network** | "One command, no manual intervention" has to hold on a grader's machine. |

---

## 1. Core idea

**The batch stream is a pure function; the ledger records its results.**

```
plan(branch_id, step) -> BatchSpec         # pure: seed x branch x step x schedule x registry
BatchSpec -> pack -> mask -> batch_hash    # sha256(input_ids, loss_mask, position_ids, segment_ids)
ledger.append(batch_committed{...})        # facts recorded after consumption, not intentions before
```

Everything the assignment asks for falls out of this:

- **Resume** = re-enter `plan()` at the checkpoint's ledger offset.
- **Replay** = re-run `plan()` over a historical interval, recompute hashes from the immutable
  shards, compare field-by-field against the ledger.
- **Fork** = same checkpoint, new `branch_id` -> a different but equally reproducible stream.
- **No skipped/repeated batches** = structural, because offsets are assigned by the plan, not by
  wall-clock ordering. Verified independently by a ledger contiguity scan.

**Consequence worth stating up front:** OPUS scoring must also be deterministic, or replay breaks.
The proxy score is derived from `sha256(candidate content) -> uniform` combined with lane/stage/
model-age priors — never from an RNG that advances with training. This is an explicit test.

**Second consequence:** if the whole demo ran in one process, "expected == actual" on resume would
be tautological. So `run_demo.py` runs the training phases as **subprocesses**, and the crash is a
hard `os._exit(1)` that destroys in-memory state. The resuming process reconstructs *only* from
the checkpoint file plus the ledger on disk. That is what makes the resume proof mean something.

---

## 2. Module layout

```
tds/
  hashing.py     canonical JSON (sorted keys, no whitespace drift) + sha256 helpers
  corpus.py      S4 shard slices + generated lanes -> docs with provenance
  tokenizer.py   frozen tokenizer wrapper, tokenizer_hash, special-token policy
  shards.py      immutable tokenized shards; writer refuses to overwrite an existing shard_id
  manifest.py    widget-6 manifest schema + two-tier admission gate
  firewall.py    4 independent checks -> rejection clauses (widget 13)
  mixture.py     stages, lane weights, protected floors, anneal reserve, supply table (widget 7)
  opus.py        accept / reject / defer / protected + decision records (widget 10)
  sampler.py     plan(branch_id, step) -> BatchSpec
  packing.py     5 policies + utilization / unused / boundary risk (widget 5)
  masks.py       per-lane loss + attention + position ids, packed_reset_on_eos (widget 3)
  model.py       tiny transformer, per-token loss, NumPy forward+backward
  trainer.py     train loop; global_step <-> ledger_offset
  ledger.py      append-only JSONL consumption ledger (widget 8 events)
  checkpoint.py  model + optimizer + dataloader offset + rng state
  recovery.py    crash / resume / replay / fork; mode: ledger | fork | random (widget 14)
  learning.py    per-shard usefulness + v6_policy_hint (widgets 11, 12)
  perf.py        4-way token fate: useful / opus-rejected / padding / loader-wait (widget 15)
  audit.py       standalone verifier; reads submission_artifacts/ + shards/ only
  evidence.py    builds evidence.json / evidence.md from the audit's findings
run_demo.py      the one command
tests/           invariant tests (pytest)
submission_artifacts/   generated
```

---

## 3. Subsystem design

### 3.1 Shards and manifests

Each shard is three files: `<shard_id>.tokens.bin` (uint32 LE), `<shard_id>.docs.jsonl`
(doc_id, token_start, token_end, lane, source provenance), `<shard_id>.manifest.json`.
Immutability is enforced by the writer (refuses to overwrite) and verified by `content_hash` over
the `.bin` bytes.

Manifest carries widget 6's exact field names: `shard_id`, `capability_lane`, `token_count`,
`tokenizer_hash`, `content_hash`, `cleaning_pipeline_hash`, `dedup_status`, `pii_screen_status`,
`eval_overlap_status`, `license_tier`, `parent_manifest_ids`, `admission` — plus `doc_count`,
`created_at`, `source_manifest`, `normalizer_id`, `special_tokens`.

**Two-tier admission gate** (widget 6's non-obvious finding — block vs. hold, not one pass/fail):

| Tier | Fields | Verdict when missing/failed |
|---|---|---|
| Hard | `pii_screen_status`, `eval_overlap_status`, `tokenizer_hash`, `content_hash` | `blocked_from_training` |
| Soft | `dedup_status`, `license_tier`, `parent_manifest_ids` | `held_for_review` |
| — | all present | `admitted_to_registry` |

Only `admitted` shards are visible to the sampler. `held_for_review` shards are stored and
manifested but never reachable by `plan()`.

**Widget 1's field checklist splits across two homes**: shard-level fields (`tokenizer_sha`,
`normalizer_id`, `source_manifest`, `capability_lane`, `pipeline_hash`, `dedup_cluster`,
`contam_scan`) live in the manifest; batch-level fields (`loss_mask_hash`, `eos_boundary`,
`position_policy`, `stage_id`, `opus_decision`, `lane_quota`) live in the ledger event. The audit
asserts every one of the 15 has a home and is populated.

### 3.2 Eval firewall

Four **independent** checks, each contributing its own clause to the rejection reason (widget 13's
point is that a single boolean is not auditable):

1. registry `never_train` flag
2. n-gram overlap against the eval fingerprint set (reusing S4's 13-gram fingerprint approach and
   its `eval_fingerprint_report.json` for MMLU / GSM8K / HellaSwag / MILU)
3. canary string match
4. benchmark-derived-content heuristic

The case worth reproducing deliberately: a doc whose registry flag says `trainable` that is still
blocked on clauses 2+4 (widget 13's `gsm8k-rationale-blog`). Registry flags alone do not protect.

**Runtime firewall** is a second, independent assertion at batch level: every token span in every
committed batch must resolve to an admitted, non-eval shard. Any eval doc id appearing at a
loss-bearing position is a hard failure of the run, not a warning.

### 3.3 Mixture compiler

Reads S5's `ledger.json` for lane weights, protected floors and anneal reserve; collapses S5's
five-marker curriculum into widget 7's three stages (`foundation` / `skill_build` / `anneal`) at
toy budget. Emits `mixture_schedule.json` and a supply table per lane:

```
planned_tokens | required_after_opus = planned / (1 - reject_rate) | verified_supply | status
```

Expect an **agentic shortfall** — widget 7 found agentic short under every profile, and our
generated agentic lane is deliberately the smallest. That's a real reproduced finding, not a
rigged one, and the compiler emits widget 7's warning-string shape for it.

### 3.4 OPUS

A candidate is a proposed batch (lane, stage, shard ids, `effective_token_estimate`). Decision
record uses widget 10's exact schema and taxonomy: `stage_mismatch`,
`duplicate_update_direction`, `eval_firewall_overlap`, `below_proxy_threshold`, `lane_quota_full`;
`deferred_for_anneal` for defers; `protected_floor_override` for protects; `null` for accepts.

**Invariant encoded and tested**: the protected floor rescues score- and quota-type rejections but
**never** overrides an `eval_firewall_overlap` hit. Floors and the firewall are independent gates
and the firewall always wins.

### 3.5 Packing and masks

Policy per lane, defended by widget 5's utilization table:

| Lane | Policy | Rationale |
|---|---|---|
| web, indic | concat-and-chop with EOS | plain pretraining; highest utilization, boundaries carried by EOS |
| code | best-fit | fills tightest window first, keeps files intact where it can |
| reasoning, agentic | structure-preserving | protects sample boundaries; widget 5 showed this cost nothing on its doc distribution |

All five policies are implemented (pad-each-doc and greedy included) so the packing report can
show the utilization/boundary-risk trade-off on our own corpus, in widget 5's table shape.

Masks per widget 3: plain lanes are fully loss-bearing on non-pad positions; the agentic lane
masks user turns and tool observations so loss-bearing < non-pad. Attention is blocked across
segment boundaries; position ids reset at EOS — recorded in the ledger as
`position_policy: "packed_reset_on_eos"` and asserted by a test.

### 3.6 Ledger and checkpoints

Append-only JSONL. `global_step` <-> `ledger_offset` maintained by the plan. Events use widget 8's
schemas verbatim, with `batch_hash`, `packing_utilization` and `loss_bearing_tokens` added:

- `batch_committed` — incl. `packed_sample_ids`, `shard_ids`, `token_span_ids`
  (`<lane>_<step>_<mb>:<start>-<end>`), `loss_mask_hash`, `position_policy`, `mixture_lane`,
  `curriculum_stage`, `opus_decision_id`
- `checkpoint_bound` — `dataloader_state: "ledger_offset_<k>"`, `rng_state`, model/optimizer state
- `worker_crash_recovered` — `failed_rank`, `recovery_mode`, `next_expected_offset`

Plus widget 14's run-level vocabulary: `run_started`, `trainer_crashed`, `checkpoint_restored`
with `mode` in {`ledger`, `fork`, `random`} and a `branch_id` that changes **only** on fork.

### 3.7 Crash, resume, replay, fork

- **Crash**: child process hard-exits mid-step at a configured step. In-memory state is lost.
- **Resume**: fresh process loads the checkpoint, reads the ledger's last committed offset,
  computes `expected = plan(branch, offset+1)`, trains it, asserts `actual.batch_hash == expected`.
  Separately scans the full ledger for contiguity and duplicate offsets. -> `resume_report.json`
- **Replay**: re-derives BatchSpecs for a historical `[step_a, step_b]` interval and recomputes
  hashes from the immutable shard bytes, comparing batch ids, token span ids, loss-mask hashes and
  batch hashes field-by-field to the recorded events. -> `replay_report.json`
- **Fork**: restores an earlier checkpoint under `branch_id: run-b`, recording parent branch and
  parent step. Stream diverges; lineage is auditable.
- **`random` mode** is run as the **negative control**: same checkpoint, no ledger, sampler
  reseeded -> a different stream. This is what proves the ledger is doing the work in ledger mode,
  and it reproduces widget 14's three-way comparison. -> a three-way table in `evidence.md`.

### 3.8 Learning ledger

Per-token loss from the model aggregates to (shard, lane, stage). First-vs-last exposure gives a
loss delta; combined with hot-token share and OPUS score it yields a usefulness score (0-100) and
a `v6_policy_hint` (`keep_with_phase_guard` / `delay_or_reclean`), matching widget 12's report-card
shape. The event chain is explicit and joinable, per widget 12's backlinks:
`batch_committed` -> `token_ppl_aggregated` -> `learning_delta_attached` -> `v6_policy_hint`.

Widget 11's finding (Indic stays the hardest lane even late) is a prediction our own trace can be
checked against — we report what we measure either way.

### 3.9 Performance

Widget 15's 4-way token fate, all **measured**, none assumed:

| Bucket | Measurement |
|---|---|
| useful | loss-bearing tokens actually trained on |
| opus_rejected | tokens prepared then rejected by OPUS |
| padding_waste | pad positions + masked-context positions |
| loader_wait | wall time in data prep vs. total step time |

Headline metric is **useful loss-bearing tokens/sec**, plus packing utilization. Every number in
`performance.json` is reconstructible from the ledger — the audit recomputes them.

### 3.10 Audit and evidence

`audit.py` is a **standalone entry point that reads only `submission_artifacts/` and the shard
files** — it does not import the trainer's in-memory state. It re-derives:

- shard content hashes from the raw bytes; tokenizer hash from the tokenizer file
- every manifest's admission verdict from its own fields
- ledger offset contiguity, monotonicity, no duplicates, append-only
- **batch hashes recomputed from shard bytes + recorded token spans**
- planned-vs-actual mixture shares and floor compliance
- firewall leak scan across every committed batch
- throughput arithmetic

`evidence.json` / `evidence.md` are generated **from the audit's findings**, so they structurally
cannot be hardcoded — the code path that writes a PASS is the code path that verified it.
`evidence.md` renders exactly the assignment's nine rows: tokenizer integrity, evaluation
firewall, packing correctness, mixture compliance, OPUS audit trail, crash recovery, replay,
learning trace, throughput.

---

## 4. `run_demo.py` sequence

One command, mapped to the 13 events `run.log` must contain:

| # | Phase | Log event | PASS marker |
|---|---|---|---|
| 1 | build corpus, tokenize, write shards | shards created | |
| 2 | validate manifests, run admission gate | manifests validated | `tokenizer_hash_verified` |
| 3 | eval firewall scan | evaluation data blocked | `eval_shard_blocked` |
| 4 | compile mixture schedule + supply table | mixture compiled | |
| 5 | OPUS scoring over candidates | OPUS decisions recorded | |
| 6 | train `run-a`, packing + ledger + checkpoints | batches packed | `checkpoint_saved` |
| 7 | hard crash mid-step (subprocess exits) | crash simulated | |
| 8 | fresh process resumes from checkpoint | run resumed | `resume_next_batch_matched` |
| 9 | replay an earlier interval | historical stream replayed | `replay_hash_matched` |
| 10 | fork to `run-b`; `random`-mode control | branch forked | |
| 11 | learning ledger rollup | | |
| 12 | performance report | performance measured | |
| 13 | standalone audit -> evidence bundle | audit completed | |

Output tree:

```
submission_artifacts/
  run.log  evidence.json  evidence.md  performance.json
  manifests/    shard manifests, mixture_schedule.json, packing_report.json
  ledgers/      consumption.jsonl, opus_decisions.jsonl, learning_ledger.json, firewall_report.json
  checkpoints/  ckpt_*.npz + checkpoint index
  reports/      resume_report.json, replay_report.json, fork_report.json, audit_report.json
```

---

## 5. Scale

~5 lanes, ~2-4k documents, ~1-3M tokens, seq len 256, global batch 8 x 4 microbatches,
~300 steps, ~1-2M-parameter model, checkpoint every 50 steps. Small enough for < 3 min on CPU,
large enough that packing utilization, mixture shares and loss deltas are meaningful rather than
degenerate.

---

## 6. Invariant tests

| Area | Test |
|---|---|
| model | finite-difference gradient check |
| tokenizer | hash frozen and verified against manifests |
| shards | writer refuses to overwrite; content hash matches bytes |
| admission | hard tier blocks, soft tier holds, complete admits |
| firewall | a `trainable`-flagged doc still blocked on overlap + derived clauses |
| firewall | no eval doc appears at a loss-bearing position in any committed batch |
| packing | reported utilization matches recomputation; no cross-segment attention; positions reset on EOS |
| masks | agentic loss-bearing < non-pad; plain lanes loss-bearing == non-pad |
| OPUS | same candidate -> same decision (determinism); protected floor never overrides firewall |
| mixture | per-stage floors honored; shortfall detected where supply < required-after-OPUS |
| ledger | append-only; offsets contiguous; no duplicates |
| resume | fresh-process next batch hash matches the expected batch |
| replay | every field matches across a historical interval |
| fork | branch id changes, stream diverges, lineage recorded |
| evidence | every claim in `evidence.json` resolves to an artifact that exists |

---

## 7. Build order

- **Phase 0** — repo init, config, `hashing.py`, determinism policy, `requirements.txt`
- **Phase 1** — corpus, tokenizer, shards, manifests, admission gate, firewall
- **Phase 2** — mixture compiler, OPUS, sampler (`plan()`)
- **Phase 3** — packing policies, masks, batch hashing
- **Phase 4** — model, trainer, consumption ledger, checkpoints
- **Phase 5** — crash / resume / replay / fork / random control
- **Phase 6** — learning ledger, performance
- **Phase 7** — audit, evidence bundle, `run_demo.py` wiring
- **Phase 8** — tests, README, graded run of record, push

Phases 1-5 carry ~700 of the 1,000 points; 6-8 carry the rest but also protect phases 1-5 from
losing credit to unverifiable evidence.

---

## 8. What changed during implementation

The plan above is kept as written. These are the places the build departed from it, and why.

**A vocabulary projection was added.** Not in the original plan. A 68,096-wide output layer
makes the logit tensor 139 MB per microbatch and the embedding table six times the rest of the
model, which would have made the demo about the model rather than the data system. The model now
trains on the 6,144 most frequent ids (~89% of token occurrences); shards, ledger events and
batch hashes still use real tokenizer ids. `tds/vocab.py`, reported in `performance.json`.

**Run resized to fit the time budget.** 300 steps at 256 context and 8,192 vocab profiled to
~32 minutes. After two optimizations (the softmax over the vocab was being computed twice --
once for the loss and once for its gradient -- and GELU's tanh was recomputed in the backward
pass, together 299ms -> 192ms per microbatch) the run is 180 steps at 192 context, finishing in
about 4.3 minutes including the crash, resume, fork, replay and audit.

**`recovery_epoch` was added to ledger events.** The plan said the crash would be mid-interval
but did not say what that does to ledger contiguity. Rolling back to step 90 after a crash at
100 means steps 90-99 are committed twice. Rather than crash on a checkpoint boundary to dodge
the problem, events carry a recovery epoch, superseded events stay in the append-only log, and
the verifier reconstructs the effective stream by taking the highest epoch per step.

**`stream_key` was separated from `branch_id`.** Widget 14's `random` mode keeps the branch id
`run-a` while re-seeding the sampler -- that is the whole hazard it illustrates -- so the stream's
hash key had to be separable from the branch it claims to belong to.

**The learning ledger's stage trend is a list, not a dict.** Artifacts are written with sorted
keys, so a dict keyed by stage name comes back alphabetically (anneal, foundation, skill_build)
and a reader would draw the curve backwards.

**Two OPUS rejection reasons are unreachable in the run of record**, covered by unit tests
instead: `stage_mismatch` cannot fire because protected floors require every lane to have
positive weight in every stage, and `eval_firewall_overlap` does not fire at batch time because
the shard-level firewall already removed those documents. The batch-time check is defence in
depth and the audit reports that it stayed clean. The other four outcomes -- accepted, rejected,
deferred and one genuine `protected_floor_override` -- all occur naturally.
