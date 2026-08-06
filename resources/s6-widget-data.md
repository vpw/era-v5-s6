# S6 Widget Data Extraction

Source lesson: https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/sessions/cms8h12vd0k9f09mmw0xb0f2f/lesson

15 widgets, standalone pages at https://axiom.theschoolofai.in/widgets/s6_widget_N_<name>.html

## Widget 1 — Requirements Ledger (`s6_widget_1_requirements_ledger.html`)

Section 1 ("What This Session Is"). Click a previous session (1-5) to see the contract it
imposes on the training dataset. Toggle production requirements to watch the admission score
change.

Header stat: **Current stream readiness: 72%** — "Blocked: core contracts absent" (this value
did not visibly change when clicking the two off toggles below; may require different
interaction than click, or may be a fixed illustrative default).

Per-session contract panels (session card -> data-system contract fields + status):

- **Session 1: Transformer Foundations** — "The model consumes fixed token windows and learns
  next-token prediction. Session 6 must preserve sequence length, ordering, loss masks,
  attention policy and EOS boundaries." Tags: next-token loss, sequence length, attention masks,
  position ids.
  - `loss_mask_hash` — Loss-bearing tokens are explicit — **OK**
  - `eos_boundary` — Document breaks survive packing — **OK**
  - `position_policy` — RoPE/context policy recorded — **WARN**

- **Session 2: Tokenization & Vocabulary Design** — "A training shard is only meaningful under
  the exact tokenizer that created it, including special tokens and Indic normalization
  choices." Tags: tokenizer hash, special tokens, Indic-safe normalization, vocab version.
  - `tokenizer_sha` — Frozen vocab and merges — **OK**
  - `special_tokens` — PAD/EOS/BOS policy stored — **OK**
  - `normalizer_id` — Unicode policy captured — **WARN**

- **Session 3: Data Collection & Sourcing** — "Every token needs provenance. Source tier,
  license status, capability tags and held-out status travel into the shard manifest." Tags:
  source id, license tier, capability tags, held-out flag.
  - `source_manifest` — URL/version/license lineage — **OK**
  - `capability_lane` — Reasoning/code/Indic/etc. — **OK**
  - `eval_firewall` — Never-train status checked — **WARN**

- **Session 4: Data Cleaning & Deduplication** — "Only cleaned, deduped, PII-screened and
  contamination-scanned data can enter the stream. The transformation lineage must be
  reproducible." Tags: cleaning hash, MinHash cluster, PII status, contamination scan.
  - `pipeline_hash` — Cleaning version attached — **OK**
  - `dedup_cluster` — Near-duplicate group known — **OK**
  - `contam_scan` — Eval overlap status present — **OK**

- **Session 5: Data Mixtures & Curriculum** — "The mixture recipe becomes an executable
  schedule: protected floors, OPUS policy, curriculum stages, anneal reserve and lane quotas."
  Tags: mixture schedule, OPUS decision id, protected floors, anneal reserve.
  - `stage_id` — Token range maps to curriculum — **OK**
  - `opus_decision` — Accepted/rejected tracked — **MISS**
  - `lane_quota` — Protected floors enforced — **OK**

Admission Toggles (checks the Session 6 pipeline must enforce before a shard enters the
training stream) — on by default: Tokenizer frozen, Eval firewall, Loss-mask hashes,
Contamination scan, Source license tier. Off by default: **OPUS reject log**, **Ledger branch
id** (clicking these did not visibly move the 72% readiness score in this session).

**Takeaway for the assignment**: this widget is effectively a checklist showing that the
Session 6 system must reproduce/verify: loss_mask_hash, eos_boundary, position_policy,
tokenizer_sha, special_tokens, normalizer_id, source_manifest, capability_lane, eval_firewall,
pipeline_hash, dedup_cluster, contam_scan, stage_id, opus_decision, lane_quota — a near-exact
manifest field list matching the session prose's shard-manifest section.

## Widget 2 — Batch Builder (`s6_widget_2_batch_builder.html`)

Section 2 (Vocabulary of a Training Step). Sliders: GPUs, Microbatch/GPU, Gradient
accumulation, Sequence length, Checkpoint interval, Toy dataset tokens (B).

Default state (matches the prose's worked example exactly):
- 8 GPUs x 2 microbatch x 16 accum = **256 global batch**
- Sequence length 4096 -> **8,192 tokens per GPU microbatch**, **1.0M tokens per optimizer step**
  (256 x 4096... wait widget shows 8,192 tokens/microbatch at seq len 4096 meaning microbatch=2
  sequences of 4096 = 8192 tokens/GPU-microbatch; global 1.0M = 256 x 4096)
- Checkpoint interval 500 steps -> **524.3M tokens between checkpoints**
- Toy dataset 5B tokens -> **4,769 steps per toy epoch**
- "One Optimizer Step Timeline" bar: accum 1 = 65,536 tokens; accum 8 = 524,288 tokens;
  accum 16 = 1.0M tokens (i.e. 8 GPUs x 4096 x accum_n).

Definitions shown: Microbatch = one small batch processed by one GPU before gradients are
accumulated. Global batch = all sequences whose gradients contribute to one optimizer update.
Checkpoint step = a saved point tying model, optimizer and data position to a global step.

Useful for sizing the toy demo's own GPU/microbatch/accum/seq-len/checkpoint-interval constants.

## Widget 3 — Document To Batch Transformer (`s6_widget_3_document_to_batch.html`)

Section 3. Choose a document type (General Web / Code / Indic / Agentic), toggle "Insert EOS
boundaries" / "Apply loss mask policy" / "Pack into 24-token window" (all on by default), watch
text -> token ids -> sequence (EOS/positions/loss mask) -> microbatch.

Fixed packed window size in this widget: **24 tokens**.

Per-lane sample documents and stats:
- **General Web**: "Climate policy changed after the monsoon report because farmers needed
  warnings" -> 12 non-pad tokens, **12 loss-bearing** (all loss-bearing — plain pretraining),
  boundary marker yes, capability lane `web`.
- **Code**: `def train ( loader ) : for batch in loader : loss . backward ( )` -> 17 non-pad
  tokens, **17 loss-bearing** (all), lane `code`.
- **Indic**: "भारत में भाषा और तकनीक का मिलन तेजी से बढ़ रहा है" -> 13 non-pad tokens, **13
  loss-bearing** (all), lane `indic`.
- **Agentic**: "user asks plan tool_call search observation results assistant compares final
  answer" -> 12 non-pad tokens, **only 10 loss-bearing** (2 tokens masked as context) — the
  only lane where loss-bearing < non-pad, visually the leading "user"/tool-observation-ish
  tokens render in red (context/no-loss) vs. blue (loss-bearing), confirming the prose's rule
  that agentic context (user request, tool observations) is masked while the model's own
  planning/tool-call/response tokens receive loss. Lane `agent`.

Good concrete evidence for how loss masks should differ by capability lane in the toy demo.

## Widget 4 — Padding Lab (`s6_widget_4_padding_lab.html`)

Section 4. Sliders: Context length (default 32), Document length (default 18). Policy buttons:
Right padding, Left padding, Crop overflow, Fill with next doc.

At context=32, doc_len=18 (14 tokens short):
- **Right padding**: 56% useful positions, 14 pad tokens, 0 cropped, 44% wasted. "Right padding
  is common for simple batches. It is easy to mask, but short documents waste long-context
  compute."
- **Left padding**: same 56%/14/0/44% split (padding side doesn't change the ratio). "Left
  padding appears often in inference because final positions align. It is usually less natural
  for pretraining pipelines."
- **Crop overflow**: same 56%/14/0/44% (doc is shorter than context here so nothing to crop;
  the label is descriptive, not numerically different at this slider setting). "Cropping can be
  fine for plain pretraining spans when boundaries are handled. It is risky for code, SFT,
  reasoning traces and agentic trajectories."
- **Fill with next doc**: **100% useful positions, 0 pad tokens, 0 cropped, 0% wasted** — filling
  the remaining 14 slots with the start of the next document eliminates padding entirely.
  "Filling with another document is normal for pretraining when EOS marks the boundary. For
  structured samples, attention and loss masks must protect the sample structure."

Clear quantitative case for why the toy demo's packer should default to concatenate-and-chop
(or best-fit packing) for plain-text lanes rather than pad-only.

## Widget 5 — Packing Simulator (`s6_widget_5_packing_simulator.html`)

Section 5. Sliders: Context length (default 64), Number of docs (default 10). Policy buttons:
Pad each doc, Concat and chop, Greedy pack, Best-fit pack, Structure-preserving.

Fixed doc-length set used for every policy (10 docs): 21, 38, 8, 25, 42, 12, 29, 46, 16, 33
tokens (sum = 270 tokens).

| Policy | Utilization | Sequences | Unused positions | Boundary risk |
|---|---|---|---|---|
| Pad each doc | 42% | 10 | 370 | none |
| Concat and chop | 70% | 6 | 114 | high |
| Greedy pack | 84% | 5 | 50 | medium |
| Best-fit pack | 84% | 5 | 50 | medium |
| Structure-preserving | 84% | 5 | 50 | **low** |

Callouts:
- Pad each doc: "is simple and boundary-safe, but it creates many partly empty windows."
- Concat and chop: "is efficient for plain pretraining. EOS markers are essential so unrelated
  documents do not look connected."
- Greedy pack: "improves utilization without much machinery. Some holes remain because document
  order is preserved."
- Best-fit pack: "usually gives higher utilization by filling the tightest available window
  first."
- Structure-preserving: "sacrifices some utilization to protect SFT, reasoning or agentic sample
  boundaries." (Interesting: in this particular doc set it actually matches best-fit's 84%
  utilization exactly while dropping boundary risk from medium to low — i.e. structure
  preservation was free here, not a tradeoff, given the specific doc-length distribution used.)

Strong reusable numbers for defending a packing-policy choice per lane in the assignment's
README (web/code -> concat-and-chop or best-fit; agentic/SFT/reasoning -> structure-preserving).

## Widget 6 — Shard Manifest Builder (`s6_widget_6_shard_manifest_builder.html`)

Section 6. Build a candidate shard: capability lane (Indic/Code/Reasoning/Agentic/General web
dropdown), token count slider, license tier (Verified commercial safe / Needs legal review /
Unknown or blocked), checkboxes (Tokenizer hash, Cleaning hash, Dedup passed, No eval overlap,
PII screened, Parent manifest). "Generate variant" randomizes a candidate shard and shows a
0-100 admission score + one of three verdicts, driven directly by which manifest fields are
present/missing:

- **87, Admitted to registry** (default `v5_indic_shard_128`, 128M tokens, Indic, license
  "safe", all fields present except `parent_manifest_ids` missing) — "The shard has enough
  metadata to be scheduled, replayed, and audited."
- **76, Held for review** (`v5_reasoning_shard_256`, 256M tokens, Reasoning, license "review",
  `dedup_status` missing) — "The shard is close, but review-only fields weaken
  reproducibility."
- **89, Admitted to registry** (`v5_agentic_shard_128`, 128M tokens, Agentic, license "review",
  all fields present) — shows license "review" alone doesn't block admission if every other
  field is present.
- **52, Blocked from training** (`v5_agentic_shard_384`, 384M tokens, Agentic, license
  "unsafe", `pii_screen_status` missing AND `eval_overlap_status` "blocked_or_unknown") — "A
  hard requirement is missing. This shard can be stored, and the trainer is blocked from
  consuming it."

Live manifest JSON schema (exact field names/shape to mirror in the toy demo's own manifest
builder):
```json
{
  "shard_id": "v5_indic_shard_128",
  "capability_lane": "Indic",
  "token_count": 128000000,
  "tokenizer_hash": "tok_4d4543e296a4",
  "content_hash": "sha256_9668bd19a4d9",
  "cleaning_pipeline_hash": "clean_a1f0f9c8122a",
  "dedup_status": "passed",
  "pii_screen_status": "screened",
  "eval_overlap_status": "clear",
  "license_tier": "safe",
  "parent_manifest_ids": [],
  "admission": "Admitted to registry"
}
```

Teaching point (verbatim): "the manifest is the contract between Sessions 1-5 and the
dataloader. If the shard is replayed later, the same tokenizer, source lineage, and safety
state must still be knowable." Key insight for the assignment: admission is gated on
**hard requirements** (pii_screen_status, eval_overlap_status — missing either -> Blocked) vs.
**soft/review fields** (dedup_status, license_tier — missing/weak -> Held for review, not
Blocked). This two-tier gate (block vs. hold) is worth replicating in the toy demo's own
admission gate rather than a single pass/fail.

## Widget 7 — Mixture Timeline Compiler (`s6_widget_7_mixture_timeline_compiler.html`)

Section 7. Controls: Total budget (120B tokens, fixed in all captures), Curriculum profile
(4-option dropdown), Warmup band (4%), OPUS rejection rate (18%). Compiles into a 3-stage
timeline (Foundation / Skill build / Anneal) with per-lane share %, tokens, and a supply-check
table (required-after-OPUS vs. verified-supply vs. satisfied/shortfall).

Fixed 3-stage timeline boundaries in every profile: **Foundation 0B-66B, Skill build 66B-102B,
Anneal 102B-120B** (of the 120B total budget).

Required-after-OPUS is computed as lane_tokens / (1 - opus_rejection_rate) = lane_tokens / 0.82.
Verified supply per lane is constant across profiles: **General 95B, Code 38B, Indic 18B,
Reasoning 22B, Agentic 10B** — this is the widget's synthetic "what we actually have" ceiling.

| Profile | General | Code | Indic | Reasoning | Agentic | Shortfalls |
|---|---|---|---|---|---|---|
| Balanced V5 (default) | 45% / 54.0B | 20% / 24.0B | 12% / 14.4B | 15% / 18.0B | 8% / 9.6B | Agentic only (11.7B req > 10B supply) |
| Code heavy | 34% / 40.8B | 34% / 40.8B | 10% / 12.0B | 14% / 16.8B | 8% / 9.6B | Code (49.8B > 38B), Agentic (11.7B > 10B) |
| Indic protected | 38% / 45.6B | 18% / 21.6B | 22% / 26.4B | 14% / 16.8B | 8% / 9.6B | Indic (32.2B > 18B), Agentic (11.7B > 10B) |
| Late anneal reserve | 28% / 33.6B | 21% / 25.2B | 14% / 16.8B | 24% / 28.8B | 13% / 15.6B | Indic (20.5B > 18B), Reasoning (35.1B > 22B), Agentic (19.0B > 10B) |

Static callouts: "Protected floors — Indic, agentic, reasoning lanes cannot be starved." /
"Anneal reserve — Strong scarce shards held for late model state." Compiler warning text
pattern: "Compiler warning: <Lane, Lane> supply is not enough after OPUS rejection. Lower
share, collect more, use repetition deliberately, or protect only the highest-value subset."

**Key structural finding for the assignment**: Agentic is the one lane that runs short under
every single curriculum profile tested (even the default, most conservative "Balanced V5") —
the widget is making the point that agentic data is the hard supply constraint regardless of
how the rest of the mixture is sliced. The toy demo's own OPUS/mixture-compliance report should
be able to reproduce this same "planned share vs. verified/actual supply, flag shortfalls"
table shape.

## Widget 8 — Training Consumption Ledger (`s6_widget_8_training_consumption_ledger.html`)

Section 8. Buttons: Commit batch, Auto run, Bind checkpoint, Simulate crash. Slider:
Microbatches per step (default 4). Header stats: global step, ledger offset, checkpoint,
branch id (default `run-a`). Each click on "Commit batch" appends one `batch_committed` event
tagged with a randomly chosen mixture lane (general/code/indic/reasoning/agentic seen).

**This is the single most valuable widget for the assignment — it gives the exact ledger event
JSON schemas to replicate.**

`batch_committed` event:
```json
{
  "event": "batch_committed",
  "ledger_offset": 3,
  "run_branch_id": "run-a",
  "global_step": 3,
  "checkpoint_id": "none",
  "created_at": "2026-08-06T17:00:10.826Z",
  "rank": 3,
  "microbatch_count": 4,
  "packed_sample_ids": ["sample_0_080f47b5", "sample_1_85466870", "sample_2_986c45fc", "sample_3_4b712ea2"],
  "shard_ids": ["shard_indic_b1199aaa", "shard_mix_74a252d5"],
  "token_span_ids": ["indic_3_0:0-4095", "indic_3_1:0-4095", "indic_3_2:0-4095", "indic_3_3:0-4095"],
  "loss_mask_hash": "lossmask_56b4985d",
  "position_policy": "packed_reset_on_eos",
  "mixture_lane": "indic",
  "curriculum_stage": "foundation",
  "opus_decision_id": "opus_cc5afb28"
}
```

`checkpoint_bound` event (fired by "Bind checkpoint"):
```json
{
  "event": "checkpoint_bound",
  "ledger_offset": 7,
  "run_branch_id": "run-a",
  "global_step": 4,
  "checkpoint_id": "ckpt_00004",
  "created_at": "2026-08-06T17:00:56.023Z",
  "model_state": "saved",
  "optimizer_state": "saved",
  "dataloader_state": "ledger_offset_6",
  "rng_state": "captured_where_available"
}
```

`worker_crash_recovered` event (fired by "Simulate crash"):
```json
{
  "event": "worker_crash_recovered",
  "ledger_offset": 6,
  "run_branch_id": "run-a",
  "global_step": 4,
  "checkpoint_id": "ckpt_00003",
  "created_at": "2026-08-06T17:00:17.066Z",
  "failed_rank": 4,
  "recovery_mode": "resume_from_last_committed_offset",
  "next_expected_offset": 6
}
```

Note token span id convention: `"<lane>_<step>_<microbatch_idx>:0-4095"` — a compact way to
carry lane + step + microbatch + token range in one id, worth copying directly. rank field
inside batch_committed events is observed equal to the microbatch index within that step (not
a fixed GPU rank across steps) in this widget's simulation — treat as illustrative rather than
literal.

Callout: "The ledger stores facts after consumption, not just intentions before training."

## Widget 9 — Checkpoint Comparison Lab (`s6_widget_9_checkpoint_comparison_lab.html`)

Section 9. Sliders: Rollback checkpoint (step 4200), Data drift without ledger (32%), OPUS
policy change (18%). Button: Run comparison. Shows two side-by-side branches from the same
checkpoint: **Run A "No ledger" (loader samples again)** — a red loss curve with a randomized
lane-tag sequence that changes on every "Run comparison" click (observed: `code code reason
agent agent opus test anneal`, then `web code code agent agent opus test anneal`) — vs. **Run B
"Ledger backed" (replay or explicit fork)** — a green loss curve with a *fixed, reproducible*
lane-tag sequence every time: `web code indic reason agent opus test anneal`.

Fixed stat tiles (did not change across reruns or slider defaults observed): **50% comparison
confidence without ledger**, **100% stream identity with replay**, **4200 checkpoint step**.

Callout: "The two branches changed model state and data stream together. A loss delta cannot be
attributed to the strategy alone." This is the widget's dramatization of the session's core
rule `experiment = model checkpoint + optimizer state + data stream + code/config` — directly
motivates why the toy demo's replay must reproduce an *identical* lane/shard sequence, not just
a statistically similar one.

## Widget 10 — OPUS Accepted And Rejected Board (`s6_widget_10_opus_audit_board.html`)

Section 10. Controls: Accept threshold slider (default 0.62), Protected floor dropdown (Indic
protected / Agentic protected / Reasoning protected / No override), Model age dropdown (early
800M tokens / mid 5.6B tokens / late 92B tokens / anneal 116B tokens), "Score next candidate"
button. Each click scores one random candidate batch (random lane, random proxy score) and
routes it into one of 4 ledgers: accepted, rejected, deferred, protected.

**Full decision-record JSON schema** (identical shape across all 4 outcomes, only
`decision`/`rejection_reason` differ):
```json
{
  "opus_decision_id": "opus_0026",
  "candidate_batch_id": "cand_8c2f166f",
  "model_age": "early 800M tokens",
  "proxy_version": "opus_proxy_v5.3",
  "lane": "indic",
  "stage": "skill_build",
  "score": 0.518,
  "decision": "protected",
  "rejection_reason": "protected_floor_override",
  "shard_ids": ["shard_indic_648", "shard_mix_463"],
  "effective_token_estimate": 718467
}
```

Observed `rejection_reason` taxonomy (5 distinct values seen across ~34 scored candidates):
`stage_mismatch`, `duplicate_update_direction`, `eval_firewall_overlap`, `below_proxy_threshold`,
`lane_quota_full`. Deferred candidates use `"decision": "deferred"` with
`"rejection_reason": "deferred_for_anneal"`. Accepted candidates have `"rejection_reason": null`.

**Key mechanic worth replicating exactly**: at accept threshold 0.90 with "Indic protected"
active, one Indic candidate scoring 0.518 (below threshold) was rescued into the **protected**
ledger with reason `protected_floor_override` — but a *different* Indic candidate
(`opus_0025`) that failed on `eval_firewall_overlap` was still **rejected**, not protected. So
**the protected floor overrides score/quota-type rejections but never overrides an eval-firewall
hit** — protected floors and the eval firewall are independent gates, firewall always wins. This
is an important, non-obvious invariant to encode in the toy demo's OPUS decision logic.

Callout: "Rejection reasons matter: quota full, duplicate gradient, weak proxy utility, stage
mismatch, and firewall hits mean different things for V6 data planning."

## Widget 11 — Token Perplexity Heatmap (`s6_widget_11_token_perplexity_heatmap.html`)

Section 11. Controls: Training phase (Early/Mid/Late/Anneal), View (Token/Lang/Shard/Pos), Hot
threshold slider (default ppl 20), trace filter checkboxes (Show masked tokens at full opacity,
Highlight boundary surprises [on by default], Overlay OPUS accepted/rejected).

Shows one fixed packed batch (same token sequence at every training phase) mixing 5 lanes
separated by `<eos>`: a pretrain snippet ("The mixture schedule reserves"), a code snippet
(`def train ( ) : \n return loss / tokens`), an Indic snippet ("भारत में शिक्षा की गुणवत्ता" —
"India's education quality"), an agentic snippet ("Observation: 404 \nThought: retry with
cache"), and a reasoning/math snippet ("Lemma: if n is prime, then ..." trailing off into a
very high-perplexity `...` token representing an unfinished proof step).

**Early phase** (default): avg token loss 3.04, avg perplexity 20.8, 15 tokens above threshold
(ppl>20), 7 masked positions. Selected token भारत ("India"): loss 5.410, perplexity **223.6**,
shard `indic-2`, lane indic, position 17, loss mask on, OPUS decision accepted. Notable
per-token peaks: गुणवत्ता ("quality") ppl 753.7, शिक्षा ("education") ppl 335.3, the reasoning
`...` (trailing proof) ppl 1900.7 — the two hardest tokens in the whole batch are Indic
subword continuations and an unfinished reasoning step.
Hardness by group (avg ppl): **indic 125.9**, agentic 23.6, reasoning 23.3, code 10.9,
pretrain 7.1 — Indic is ~18x harder than plain pretrain text at this phase.

**Late phase**: avg token loss drops to 1.68, avg perplexity 5.4, only 4 tokens above threshold.
Same भारत token: loss 3.142, perplexity 23.2 (down from 223.6). Hardness by group: **indic
17.3**, reasoning 5.6, agentic 5.5, code 3.6, pretrain 2.9 — every group got easier, but Indic
is *still* ~6x harder than pretrain and remains the hardest group by a wide margin even late in
training. Order of groups by hardness stays identical (indic > reasoning/agentic > code >
pretrain) across phases; only the gap narrows.

Callout: "The trace is most useful when linked to model age. A high-perplexity token at 400M
tokens can be healthy novelty; the same token at 5.6B tokens may indicate missing data, bad
tokenization, or a fragile capability lane."

**Reusable finding for the assignment's learning-ledger section**: Indic (and the "unfinished
reasoning step" token) staying disproportionately hard even after the model matures is exactly
the kind of per-shard/per-lane signal the two-way learning ledger (session prose section 12)
should surface — it's concrete evidence for why Indic needs a protected floor and possibly more
verified native tokens rather than more repetition of the same data.

## Widget 12 — Shard Learning Report Card (`s6_widget_12_shard_learning_report_card.html`)

Section 12. Global controls: Model phase (Early/Mid/Late/Anneal), Usefulness weight slider
(default "loss delta 60%"). Left panel lists 5 example training shards, each tagged `review` or
`delay` and showing a `hot ppl` percentage. Selecting one shows: usefulness score (0-100), loss
delta, hot token share %, OPUS score (0-100), a 4-bar early/mid/late/anneal loss trend chart, a
one-line verdict, and 4 "Ledger Backlinks" entries (`batch_committed`, `token_ppl_aggregated`,
`learning_delta_attached`, `v6_policy_hint`) — i.e. the learning ledger is explicitly built by
joining these named event types.

All 5 shards, evaluated at "mid" model phase:

| Shard | Lane | Tokens | Tag | Usefulness | Loss delta | Hot ppl share | OPUS score | Loss trend (early→mid→late→anneal) | v6_policy_hint | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| indic-tier-a-042 | Indic | 84M | review | 60 | -0.22 | 24% | 82 | 3.40 → 2.60 → 2.15 → 1.72 | keep with phase guard | "Strong late learner. Preserve for anneal and expand with adjacent native text." |
| code-repair-118 | Code | 51M | review | 63 | -0.25 | 18% | 74 | 2.90 → 2.10 → 1.72 → 1.68 | keep with phase guard | "Useful in mid-training; diminishing returns in anneal." |
| agentic-browser-021 | Agentic | 19M | review | 65 | -0.28 | 33% | 68 | 4.10 → 3.35 → 2.74 → 2.21 | keep with phase guard | "Hard but productive. Keep observations masked and repeat after tool grammar stabilizes." |
| web-clean-900 | Web | 420M | **delay** | 40 | -0.11 | 11% | 48 | 2.50 → 1.84 → 1.55 → **1.60** (ticks back up in anneal) | **delay or re-clean** | "Broad base data. Valuable early, weak late; avoid spending anneal budget here." |
| reason-proof-077 | Reasoning | 12M | review | 57 | -0.20 | 38% | 91 | 4.60 → 3.80 → 2.95 → 2.04 | keep with phase guard | "Looks too hard early, becomes excellent once the base model is ready." |

Each shard's `batch_committed` backlink cites the same `step 18420, checkpoint ckpt-18` (i.e.
all 5 example shards are illustrating the same one evaluation checkpoint from different lanes).

**Reusable finding**: web-clean-900 is the one shard that gets *worse* late→anneal (1.55 ->
1.60) and is the only one tagged `delay` with a distinct `v6_policy_hint` ("delay or
re-clean") — a concrete example of the session's "if a shard causes gradient spikes /loss stops
improving, it may need cleaning, staging, or warmup" rule (prose section 12). Good template for
the toy demo's own per-shard learning-ledger verdict field (`keep_with_phase_guard` vs.
`delay_or_reclean`, etc.) rather than a plain useful/neutral/harmful enum.

## Widget 13 — Eval Firewall (`s6_widget_13_eval_firewall.html`)

Section 13. Firewall Checks (4 independent toggles, all on by default): Block shards with
never_train=true, Block MinHash/exact benchmark overlap, Block canary string matches, Block
benchmark-derived explanations. Buttons: Scan next candidate, Reset stream. A fixed 7-candidate
stream is scanned one at a time; each gets a Gate Result (Registry flag, Overlap scan %, Canary
scan, Derived data) and an Access Log line.

Fixed candidate stream (name / description / overlap%% / registry flag):
1. `train-web-104` — Web train, 18M, overlap 2% — `trainable`
2. `mmlu-mirror-3` — Benchmark mirror, 1.2M, overlap 91% — `never_train`
3. `code-sft-clean-81` — Code SFT, 420K, overlap 4% — `trainable`
4. `gsm8k-rationale-blog` — Derived explanation, 230K, overlap 32% — `trainable` (registry flag
   says trainable, but gets blocked anyway — see below)
5. `indic-news-72` — Indic train, 9M, overlap 1% — `trainable`
6. `swebench-solution-dump` — Eval solution, 88K, overlap 74% — `never_train`
7. `reasoning-forum-14` — Reasoning, 3M, overlap 18% — `trainable`

**With all 4 checks on** (default), scanning all 7: **4 admitted, 3 blocked.**
- Admitted: train-web-104, code-sft-clean-81, indic-news-72, reasoning-forum-14 — "registered
  access and admitted to train stream"
- Blocked: mmlu-mirror-3 — "rejected: never_train flag, benchmark overlap 91%, canary match";
  swebench-solution-dump — "rejected: never_train flag, benchmark overlap 74%, canary match,
  benchmark-derived content"; **gsm8k-rationale-blog** — "rejected: benchmark overlap 32%,
  benchmark-derived content" — this is the key example: its own registry flag says
  `trainable` (it is NOT marked never_train), yet the firewall still blocks it purely on the
  overlap-scan + derived-content checks. **This is the widget's core teaching point**: a shard
  can look clean on its registry flag alone and still leak eval knowledge — the firewall must
  run independent overlap/canary/derived-content scans, not just check a boolean flag.

**With "Block MinHash / exact benchmark overlap" turned off** (only 4 candidates scanned before
recording): gsm8k-rationale-blog now blocks with just "rejected: benchmark overlap 32%" (the
"benchmark-derived content" clause dropped from the reason string), and Gate Result's "Derived
data" field showed "benchmark derivative" (as opposed to "clear" on the earlier admitted
train-web-104 shard) — confirms each of the 4 checks independently contributes a clause to the
rejection reason string, and disabling one narrows (but doesn't necessarily eliminate) the
block, exactly matching the widget's own hint: "Turn checks off to see how an apparently clean
training shard can leak eval knowledge through near duplicates, synthetic explanations, or
public benchmark mirrors."

**Reusable finding**: the toy demo's eval firewall should implement (at least) these same 4
independent checks — never_train registry flag, overlap-hash match, canary string match,
derived/synthetic-explanation detection — each contributing its own clause to a rejection
reason, rather than a single pass/fail boolean, so that "how" a shard was blocked is auditable.

## Widget 14 — Crash, Replay, Fork (`s6_widget_14_crash_replay_fork.html`)

Section 14. **This is the single most directly applicable widget** — it is a working miniature
of exactly what the assignment's final demo must prove (crash -> resume -> exact next batch;
replay an earlier interval -> matching ids). Buttons: Advance 100 steps, Crash now, Rollback to
ckpt-N, Reset. Recovery Mode radio: **Ledger replay** ("Feed the same sample ids after
rollback"), **No ledger** ("Sampler restarts from seed and may feed different data"),
**Intentional fork** ("Keep checkpoint, create a new data branch"). Checkpoints auto-created
every 100 steps (ckpt-0 at step 0, ckpt-1 at step 100... observed ckpt-1 at step 300 meaning a
checkpoint cadence of ~every-100-steps up to some cap of 3 checkpoints shown). Run state tiles:
global step, ledger offset, checkpoint, branch id (default `run-a`).

Sequence run and captured (all from the same starting checkpoint history: ckpt-0 -> ckpt-1
[step 100? shown as bound at step 300] -> ckpt-2 [step 400]):

1. Advanced 3x100 steps -> step 300, offset 300, checkpoint ckpt-1, branch run-a.
   `batch_committed` log lines: "advanced to step 100 and ledger offset 100", then 200, then
   300 (offset always equals global step 1:1 in this widget's toy model).
2. **Crash now** at step 300 -> `trainer_crashed`: "failure at step 300, checkpoint ckpt-1,
   offset 300". Branches-and-failures marker flips from "current" to "crash/restore".
3. **Resume** (Advance 100 steps again, no explicit "resume" button — just continuing) -> step
   400, ledger offset 400, checkpoint auto-advances to ckpt-2. `batch_committed`: "advanced to
   step 400 and ledger offset 400".
4. **Rollback to ckpt-2** with Recovery Mode = **Ledger replay** -> `checkpoint_restored`:
   "restored ckpt-2 with mode ledger on run-a" — **Next Four Batches After Recovery is
   byte-identical to the batch list shown before the rollback**: `sample-7000/web-20/web,
   sample-7001/code-21/code, sample-7002/indic-22/indic, sample-7003/agentic-23/agentic` —
   this is the widget's concrete proof of deterministic replay.
5. Switched Recovery Mode to **Intentional fork**, then **Rollback to ckpt-2** again ->
   `checkpoint_restored`: "restored ckpt-2 with mode fork on **run-b**" (branch id changed from
   run-a to run-b) — Next Four Batches becomes a **completely different set**:
   `sample-7017/indic-24/indic, sample-7018/agentic-25/agentic, sample-7019/reason-26/reasoning,
   sample-7020/web-27/web`. Callout changes to: "Fork mode is valid when intentional: the
   branch id says this is a new data experiment from the same checkpoint."
6. Switched Recovery Mode to **No ledger**, then **Rollback to ckpt-2** again ->
   `checkpoint_restored`: "restored ckpt-2 with mode random on run-a" (branch id reverts to
   run-a, but this time via a `random` sampler) — Next Four Batches becomes **yet another
   different set**: `sample-7009/reason-29/reasoning, sample-7010/web-30/web,
   sample-7011/code-31/code, sample-7012/indic-32/indic`. Callout: "Without a ledger, the same
   checkpoint can see a different next stream. A loss curve difference now mixes model effects
   with data effects."

**Exact event-name/mode vocabulary to mirror in the toy demo's own log**: `run_started`,
`batch_committed`, `trainer_crashed`, `checkpoint_restored` with a `mode` field taking values
`ledger` / `fork` / `random`, and a `branch_id` that only changes on fork. Sample id / lane-tag
naming convention: `sample-<n>` paired with `<lane_prefix>-<counter>` (e.g. `web-20`,
`code-21`, `indic-22`, `agentic-23`, `reason-26`) — worth mirroring directly since it doubles as
both a token-span reference and a mixture-lane tag in one compact pair, same spirit as widget
8's `"<lane>_<step>_<idx>:0-4095"` token span ids.

Top-level callout: "A checkpoint captures model state. The ledger captures the consumed data
frontier. Together they decide whether an experiment is resumed, replayed, or intentionally
branched." And: "Recovery reads the ledger offset and replays the historical sample stream, so
the old checkpoint comparison stays clean."

## Widget 15 — Dataloader Throughput Lab (`s6_widget_15_dataloader_throughput_lab.html`)

Section 15. Sliders (9 knobs): sequence length (4096 tokens), global batch (256 seq), packing
efficiency (86%), OPUS rejection (18%), loader workers (18), prefetch depth (6), shard size
(1024 MB), storage bandwidth (1800 MB/s), decompression cost (24%). Header stat tiles: useful
tok/sec, GPU idle %, loader wait/step (ms). "Token Budget Per Step" shows 4 stacked bars (green
= trains the model / amber = prepared but OPUS-rejected / red = padding-or-packing waste / gray
= GPU time lost waiting) for 4 scenarios: **Current**, **No OPUS reject**, **Tighter packing**,
**Fast storage** — a direct visual answer to "where do my tokens actually go."

Default ("Current") state: **2.71M useful tok/sec, 0% GPU idle, 0 ms loader wait/step.**
Bottleneck Readout bars: packing 86%, OPUS kept 82% (= 100% - 18% rejection), bandwidth 64%,
worker cover 51%, **GPU busy 100%**. Callout: "Healthy pipeline: most step time becomes useful
training tokens."

Dragging loader workers 18->2 and storage bandwidth 1800->200 MB/s: Bottleneck Readout bars
dropped sharply (bandwidth 64%->4%, worker cover 51%->16%), but the **headline useful tok/sec,
GPU idle%, and GPU busy% tiles did not change** (stayed 2.71M / 0% / 100%) — i.e. in this
widget's implementation the top-line throughput numbers are not wired to the
worker-count/bandwidth knobs, only the Bottleneck Readout sub-bars are. Worth noting as a
caveat if citing this widget's numbers, and a useful real-world lesson in itself: **a metric
dashboard can look "healthy" (100% GPU busy) while its own sub-components show the underlying
resource is badly under-provisioned** — exactly the session's warning that "a loader may report
high raw token throughput while still wasting compute on padding, context-only tokens, or
batches rejected by OPUS."

Definitional callout: "Green tokens train the model. Amber tokens were prepared but rejected by
OPUS. Red is padding or packing waste. Gray is GPU time lost waiting for the loader." This
4-way token-fate split (useful / OPUS-rejected / padding-waste / loader-wait) is a good template
for the toy demo's own `performance.json` — track all 4 buckets, not just raw throughput.
