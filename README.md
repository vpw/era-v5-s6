# Training Data Execution System — ERA V5, Session 6

A small but complete data system for LLM training: documents → tokenized shards → manifests →
mixture schedule → packing → batches → training → consumption ledger → learning ledger →
checkpoint → crash → resume → replay → audit.

The model is a toy. The data system is the deliverable.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_demo.py
```

One command, no network, ~5 minutes on a CPU. It regenerates `submission_artifacts/` from the
committed fixtures and `config/run_config.json` alone.

```bash
.venv/bin/python -m pytest tests/ -q     # invariant tests
.venv/bin/python -m tds.audit_cli        # re-audit an existing artifacts tree
```

---

## The one idea

**The batch stream is a pure function; the ledger records its results.**

```
plan(branch_id, step) -> BatchSpec       # seed × branch × step × schedule × admitted registry
BatchSpec -> pack -> mask -> batch_hash  # sha256(input_ids, loss_mask, position_ids, segment_ids)
ledger.append(batch_committed{...})      # facts after consumption, not intentions before
```

Resume, replay and fork are then one mechanism seen three ways, rather than three features:

| | mechanism |
|---|---|
| **Resume** | re-enter `plan()` at the checkpoint's ledger offset |
| **Replay** | re-run `plan()` over a historical interval, recompute hashes from the immutable shards, compare to the ledger field by field |
| **Fork** | same checkpoint, new `branch_id` → a different but equally reproducible stream |

Two consequences fall out of this, and both shaped the implementation:

**OPUS scoring has to be deterministic.** The proxy score is derived from
`sha256(candidate content)` crossed with lane/stage/model-age priors — never from an RNG that
advances with training. If the selector drifted, replaying an interval would reach different
decisions and every hash in the ledger would stop matching. Tested in
`tests/test_opus_mixture_ledger.py`.

**The crash has to be a real process death.** `run_demo.py` runs its training phases as
subprocesses and the crash is `os._exit(1)`. The resuming process is a fresh interpreter that
rebuilds the stream from the checkpoint and ledger on disk. Had the demo run in one process,
"the next batch is the expected batch" would compare an object to itself.

---

## What comes from Sessions 1–5

The session is explicitly cumulative, so this repo consumes real upstream artifacts rather than
synthesising equivalents. Everything under `fixtures/` was copied by `scripts/build_fixtures.py`
and is committed so the repo clones and runs standalone.

| Session | Contract | How it is honoured here |
|---|---|---|
| **S1** | fixed windows, next-token loss, masks, EOS boundaries | `masks.py` — `segment_ids`, `position_ids` resetting on EOS (`packed_reset_on_eos`), and a loss mask consistent with both |
| **S2** | a shard is meaningful only under the tokenizer that made it | the frozen 68,096-entry sarvam1 vocab, hashed to `tok_bb5115a36ddb`; nothing here trains or extends a vocabulary |
| **S3** | every token needs provenance | `source_manifest`, `capability_lane`, licence tier derived from S4's verified/unverified source pools |
| **S4** | only cleaned, deduped, PII-screened, decontaminated data | the web and Indic lanes **are** S4's `06-decontaminated` output; `cleaning_pipeline_hash` is a hash over S4's nine recorded cleaning-script SHA-256s; the firewall scans S4's own cached MMLU/GSM8K/MILU sets with S4's 13-gram/2-hit rule |
| **S5** | the mixture recipe becomes an executable schedule | lane weights, protected floors, anneal reserve, epoch ceiling and the OPUS keep fraction (0.4) are read from S5's `data/ledger.json` |

Code, reasoning and agentic lanes are **generated** — S4's Sangraha corpus has none — and say so
in their provenance. Nothing pretends to be more real than it is.

---

## Layout

```
tds/
  hashing.py     canonical JSON + sha256; stable_uniform for reproducible pseudo-randomness
  config.py      one config file drives everything; its hash travels into the evidence
  corpus.py      S4 documents + generated lanes, with role-tagged segments
  tokenizer.py   the frozen vocab, hashed
  shards.py      immutable shards; the writer refuses to overwrite a shard id
  manifest.py    widget 6's schema + the two-tier admission gate
  firewall.py    four independent checks, each contributing its own rejection clause
  mixture.py     S5's recipe compiled to a per-microbatch lane plan, + supply check
  opus.py        accept / reject / defer / protect, deterministic
  sampler.py     plan(branch, step) -> BatchSpec        <- the spine
  packing.py     five policies with utilisation and boundary risk
  masks.py       loss / attention / position, per lane
  model.py       1.2M-parameter NumPy transformer, hand-written backward
  trainer.py     the loop; global_step <-> ledger_offset; loss attribution
  ledger.py      append-only JSONL, widget 8 + widget 14 event schemas
  checkpoint.py  model + optimizer + dataloader offset + config/registry identity
  recovery.py    resume, replay, fork, and the random-mode negative control
  learning.py    two-way learning ledger, per-shard usefulness and v6_policy_hint
  perf.py        four-way token fate, throughput, mixture compliance
  audit.py       nine checks, reading only the generated artifacts
  evidence.py    renders evidence.json / evidence.md from the audit
```

---

## Design decisions worth defending

**The admission gate has two tiers, not one.** Missing `pii_screen_status` or
`eval_overlap_status` blocks a shard from training outright; a missing `dedup_status` or a weak
licence only holds it for review. A single pass/fail would erase the difference between a shard
that is dangerous and one that is merely under-documented. All three verdicts occur naturally in
the run.

**Shards are grouped by (lane, metadata signature).** Admission is decided per shard, so mixing
documents with different safety metadata would force the gate to judge 220 good documents by
their worst neighbour. Segregating them means an incomplete document quarantines itself.

**The firewall runs four checks, not a flag lookup.** Its most important result is the pair of
documents flagged `trainable` in the registry that are blocked anyway, on
`benchmark overlap + benchmark-derived content` — widget 13's `gsm8k-rationale-blog` case,
reproduced against real GSM8K text. Registry flags do not protect you from a blog post that
quotes the test set.

**The mixture is compiled ahead of time into an explicit lane plan.** 720 microbatch slots,
assigned before training starts and written to `manifests/mixture_schedule.json`. This makes
planned shares exact rather than approximately right after 720 draws, keeps `plan()` pure (no
counter has to be replayed to know which lane slot 500 belongs to), and turns "mixture
compliance" into a comparison between two artifacts instead of a claim. Slots are spread by
stride scheduling — a run that trains 200 consecutive Indic microbatches is not the same
experiment as one that interleaves them.

**Rejected candidates still advance the lane cursor.** They were read, tokenized and packed
before the selector saw them, so they are spent supply. This is why the compiler sizes demand as
`planned / keep_fraction` and why OPUS-rejected tokens get their own bucket in the token-fate
report instead of being folded into "useful".

**Rollback is handled honestly rather than avoided.** The crash lands at step 100 and the last
checkpoint is at 90, so recovery rolls the model back and re-consumes steps 90–99. Those are not
repeated batches — the model state was rolled back with the data, so each (model state, batch)
pairing still happens exactly once. Every committed batch carries a `recovery_epoch`; superseded
events stay in the log (it is append-only); and the verifier reconstructs the *effective* stream
by taking the highest epoch per step before checking contiguity. Crashing neatly on a checkpoint
boundary would have made the proof easier and weaker.

**The `random` recovery mode is run as a negative control.** It has no place in a real pipeline,
which is exactly why it is here: without it, "ledger replay reproduces the stream" is
unfalsifiable. The three-way comparison shows the same checkpoint yielding three different
futures, and only ledger mode reproducing the recorded one — with the re-seeded run still
calling itself `run-a`.

**The evidence bundle is generated from the audit, not from the run.** `audit.py` imports no
trainer state. It opens `submission_artifacts/` and `work/shards/`, recomputes hashes from bytes,
re-derives every admission verdict from its manifest fields, reconstructs the effective batch
stream from the ledger, and recomputes the throughput arithmetic. `evidence.py` renders what it
found. There is no code path that writes `PASS` without a recomputation having agreed.

---

## Honest caveats

**The model's vocabulary is projected.** The output layer is 6,144 wide, not 68,096 — the top
ids by corpus frequency, covering ~89% of token occurrences. A full-vocab logit tensor would
dominate a demo whose point is the data system. The projection is kept at arm's length from the
data path: **shards, ledger events, batch hashes and replay comparisons all use real tokenizer
ids**, and the coverage figure is reported in `performance.json` so the cost is visible.

**Throughput numbers are CPU NumPy on a toy model.** The transferable figures are the ratios —
what share of prepared positions became loss-bearing, how much went to padding, how much OPUS
discarded — not the absolute rate.

**Two OPUS rejection reasons are not exercised by the run of record.** `stage_mismatch` cannot
fire because every lane has positive weight in every stage (the protected floors require it), and
`eval_firewall_overlap` does not fire at batch time because the shard-level firewall already
removed those documents — the batch-time check is defence in depth, and the audit reports that it
stayed clean. Both branches are covered by unit tests instead, including the invariant that a
protected floor never rescues a firewall hit.

**Agentic is left short on supply.** It needs ~1.5 epochs of repetition where every other lane is
covered by unique tokens. Widget 7 found agentic runs out under every curriculum profile, and
generating synthetic trajectories until the compiler warning disappeared would have hidden the
one supply constraint the session says is real.

**The generated lanes are far easier than the real ones, and the learning ledger shows it.**
Measured mean loss lands at ~6.4 for Indic and ~5.7 for web against ~0.7–1.0 for code, reasoning
and agentic. That gap is not a finding about those capabilities — it is an artifact of code,
reasoning and agentic being template-generated with a small effective vocabulary, so a 1.2M
-parameter model memorises them quickly, while real Sangraha text stays genuinely hard. The
ranking does put Indic hardest, matching widget 11, but web placing second is the template effect
rather than a result. Treat the *machinery* — loss attributed per shard, per lane, per stage,
with a policy hint attached — as the deliverable, not the loss values.

**A short run does not reach every shard.** 10 of 31 admitted shards receive loss attribution;
each lane's cursor walks its shards in order and stops where the run stops. The learning ledger
reports this coverage explicitly, because "never opened" and "judged unhelpful" are different
states and a shard with no verdict should not be read as a shard with a bad one.

---

## Two results that read as contradictions and are not

**Planned epochs vs. realised epochs.** The mixture compiler projects agentic at ~1.5 epochs,
but `performance.json` reports every lane at 0 completed epochs. Both are right. The compiler
sizes demand as `planned / keep_fraction` using **S5's planning assumption of a 60% OPUS
rejection rate**; this run's selector actually rejected 38.5%, so real demand came in well below
the conservative projection and no lane wrapped its cursor. The two figures are reported side by
side in `performance.json` (`opus.planned_keep_fraction` against `opus.rejection_rate`) precisely
so the gap is legible. A planning assumption being pessimistic is the normal case, not an error —
but it would look like one if only the projection were shown.

**Structure-preserving packing costs utilisation here, and did not in the session.** Widget 5
found structure-preserving matching best-fit at 84% on its particular document lengths, so
protecting sample boundaries was free. On our documents it is not: Indic and web reach ~99.5%
under concat-and-chop, while agentic and reasoning sit at 63–65% under structure-preserving.
Boundary safety is bought with padding, and how much it costs depends entirely on the length
distribution. That is why all five policies are implemented and measured per lane in
`manifests/packing_report.json` rather than a single number being quoted from the widget.

---

## Artifacts

```
submission_artifacts/
  run.log                     full event sequence with [PASS] markers
  evidence.json / evidence.md nine requirements, generated from the audit
  performance.json            token fate, throughput, mixture compliance
  manifests/                  per-shard manifests, shard_index, mixture_schedule,
                              packing_report, corpus_summary, vocab_projection
  ledgers/                    consumption.jsonl, opus_decisions.jsonl,
                              learning_ledger.json, firewall_report.json
  checkpoints/                ckpt_*.npz + meta + checkpoint_index
  reports/                    resume, replay, fork/divergence, audit
```

The tree committed here is the graded run of record.

**Reproducibility is verified, not asserted.** The repo was cloned to a clean directory, given a
fresh virtualenv, and run end to end. Against the committed run, the regenerated tree matched on:

| | |
|---|---|
| shard ids, content hashes, lineage hashes, token counts | identical |
| admission verdicts and per-lane supply | identical |
| all **210** committed batch hashes | identical |
| all token span ids and loss-mask hashes | identical |
| model hash at every checkpoint | identical |
| the OPUS decision log | byte-identical |

Only timestamps and wall-clock timing figures differ. That is the property the whole design
exists to produce: given the same fixtures and the same config, the data stream — and therefore
the experiment — is reconstructible by anyone.

**Checkpoint weights are not committed.** Each checkpoint is ~15 MB of float32 (model plus Adam
moments), 139 MB across the run. `submission_artifacts/checkpoints/*.npz` is gitignored; every
checkpoint's `.meta.json` and `checkpoint_index.json` **are** committed, and those carry the model
hash, state SHA-256, bound ledger offset and fork lineage. The audit and the tests read the
metadata, never the weights. `python run_demo.py` regenerates the blobs.
