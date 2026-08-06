# Session 6: Building the Training Dataset

Source: https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/sessions/cms8h12vd0k9f09mmw0xb0f2f/lesson

## 1. What This Session Is

Session 5 ended with a training data recipe. We selected: Capability buckets, Protected parts,
Curriculum stages, Annealing reserves, OPUS selection, Benchmark-backed mixture targets.

That design is useful only if the training system can execute it accurately, consistently, and
reproducibly. Session 6 converts those requirements into the actual data stream consumed by the
training loop.

A training run receives: Token windows, Loss masks, Attention masks, Position IDs, Mixture tags,
Packed microbatches.

This data must remain correct across: Many workers, Many GPUs, Training restarts, Checkpoint
boundaries.

The objective of this session is to build the system that keeps this stream controlled,
inspectable, and replayable.

The requirements come from every earlier session:

- Session 1 defined the training contract: predict next token, loss only on intended tokens,
  attention bounded to context window.
- Session 2 defined the tokenizer contract: same raw text always produces same token IDs,
  canonical special tokens, Indic-safe normalization.
- Session 3 defined the source contract: every source carries provenance, license, held-out
  status, capability tags, token accounting.
- Session 4 defined the admission contract: only cleaned, deduplicated, PII-screened,
  language-validated, contamination-scanned data may enter training.
- Session 5 defined the mixture contract: curriculum stages, protected floors, OPUS selection,
  annealing reserves, long-context timing, reasoning-length bands.

Session 6 combines all of these into one operational object: **the training data execution
system**. A requirements ledger traces every decision from Sessions 1-5 into a concrete
obligation for Session 6 (next-token loss maps, tokenizer hashes, source provenance, cleaning
manifests, evaluation firewalls, mixture floors, OPUS logs, annealing reserves).

Core principle: **the dataloader is downstream of every design choice made so far.**

The rest of the session follows one piece of data as it moves through the system:
`Cleaned text -> packed batch -> ledger event -> next corpus version`. The final step turns
what happened during training into a learning signal for the next version of the corpus.

## 2. The Vocabulary of a Training Step

- **Token**: integer ID produced by the tokenizer.
- **Sequence**: fixed-length window of tokens (4,096 / 8,192 / 32,768).
- **Sample**: one training example handed to the model. In pretraining, usually one fixed-length
  sequence. In SFT/agentic training, may contain multiple fields: prompt tokens, response
  tokens, tool observations, masks, labels.
- **Microbatch**: the small batch processed by one GPU before gradient accumulation.
- **Global batch**: complete set of samples contributing to one optimizer update across all GPUs
  and all gradient accumulation steps.
- **Training step**: one optimizer update.
- **Checkpoint step**: a training step at which enough state is saved to resume: model weights,
  optimizer state, scheduler state, RNG state (where available), dataloader state, ledger
  offset.

Worked example: 8 GPUs x microbatch size 2 x gradient accumulation 16 = 256 sequences per
optimizer step. At sequence length 8,192, that step consumes 256 x 8,192 = 2,097,152 token
positions.

Some token positions are useful loss-bearing tokens, some are padding, some are context-only.
The GPU computes over every position, but the model learns only from positions where loss is
applied.

Widget: batch builder — students control GPUs, microbatch size, gradient accumulation, sequence
length, checkpoint interval. Computes microbatch tokens, global batch tokens, optimizer steps,
consumed tokens between checkpoints, number of ledger events created.

If the ledger records that global step 843,219 consumed a batch, every microbatch and packed
sample inside that optimizer update can be reconstructed.

## 3. Documents Become Training Sequences

Humans think in documents. Training operates on token windows. A cleaned document (text +
provenance + quality metadata) moves through a fixed transformation:

```
clean document -> token IDs -> token spans -> packed sequence -> microbatch -> global batch -> optimizer step
```

The system must preserve meaning and boundaries during this transformation:

- **EOS token** may mark the end of a document.
- **Document ID** identifies the source that produced a token span.
- **Loss mask** determines whether a token contributes to the gradient.
- **Attention mask** determines which earlier tokens each token can attend to.
- **Position IDs** tell the model where each token sits inside the sequence.

Loss pattern by training mode:
- Plain pretraining: most tokens receive next-token prediction loss.
- SFT: prompt provides context; assistant response receives loss.
- Agentic: user request + tool observations are context; planning, tool calls, and final
  response receive loss.

The batch must carry more than token IDs — it must carry the training meaning of those token
IDs.

Widget: document-to-batch transformer — several documents becoming token ids, token spans,
packed sequences, microbatches, with EOS markers, document boundaries, loss masks, attention
masks, position ids and metadata ids drawn explicitly. Switching pretraining/SFT/agentic mode
changes which tokens receive loss.

This is the first place Session 5's loss maps become operational — the batch tells the
optimizer where learning should happen.

## 4. Padding and Why It Hurts

Training systems prefer fixed shapes; natural text does not arrive in fixed shapes. Padding
solves the shape problem but creates a compute problem — padded positions occupy memory, pass
through parts of the model, and reduce useful tokens processed per second.

Kinds of padding:
- **Right padding**: pad tokens after real tokens (common batching short examples).
- **Left padding**: pad tokens before real tokens (more common in batched inference).
- **Batch-level padding**: pad every example to the length of the longest example in the batch.
- **Fixed-context padding**: pad every sample to the model's full context length.
- **Within-pack holes**: unused space left at the end when packing multiple examples into one
  fixed window.

Even if the loss mask ignores pad positions, memory bandwidth and compute have already been
spent carrying them through the batch (unless the implementation can explicitly skip them). On
large runs, padding is wasted training budget.

Three questions addressed:
1. **Can we cut a document mid-line?** For plain next-token pretraining, yes — the sequence
   boundary is primarily an engineering boundary. But careless cuts can damage code blocks,
   tables, proofs, agent trajectories, and instruction-response examples where structure
   matters.
2. **Can we fill the remaining window with a different topic?** For plain pretraining, yes —
   documents are concatenated with EOS boundaries. For SFT/agentic/reasoning traces, mixing
   unrelated samples in the same attention-visible context can teach unnatural transitions
   unless masks and boundaries isolate them.
3. **Does the boundary matter if the model only predicts next token?** Yes — without explicit
   boundaries the model may learn that unrelated text is a natural continuation.

Widget: padding lab — compares right padding, left padding, batch-level padding, fixed-context
padding, within-pack holes. Shows useful-token %, wasted positions, estimated compute waste,
and whether the policy is safe for plain pretraining, SFT, agentic traces, long reasoning
examples. Includes a "half-line crop" control.

## 5. Packing Policies

Packing fills fixed-length sequences with useful tokens.

- **Pad-only**: each sample becomes one sequence, remaining space padded. Preserves structure,
  wastes compute.
- **Concatenate-and-chop**: documents joined with EOS markers, fixed-length windows cut from the
  resulting stream. Efficient for plain pretraining; treats boundaries as mechanical cut points.
- **Greedy packing**: each example placed into the first available sequence with enough
  remaining space. Improves utilization but depends on incoming order.
- **Best-fit packing**: sorts/buckets examples by length, places each into tightest available
  space. Improves utilization further, especially with many short examples.
- **Structure-preserving packing**: adds rules for SFT/tool-use/agentic data so unrelated
  examples don't leak into each other via attention.
- **Long-context packing**: handled separately — long-context batches are expensive, every
  unused position wastes a high-value training opportunity.

Policy depends on data type: plain web text tolerates concatenation; code tolerates spans but
benefits from preserving file/function boundaries; agentic trajectories must preserve tool
observation/call/answer order; reasoning traces need room to complete the argument. Evaluation
shards must never be casually packed into training.

Widget: packing simulator — choose data type + packing policy among pad-only,
concatenate-and-chop, greedy, best-fit, structure-preserving, long-context. Reports utilization,
truncation, boundary crossings, attention-mask complexity, loss-mask safety.

Every unused slot is a token the model did not learn from.

## 6. Tokenized Shards and Manifests

Tokenization happens before training and must be frozen — the training loop consumes tokenized
shards tied to one exact tokenizer version. A **shard is an immutable training object**, stored
as indexed binary token arrays (pretraining), structured records (SFT/agentic), or sharded
tar-style objects (multi-file examples). Storage format may vary by stage; manifest discipline
must stay consistent.

Shard manifest fields: Shard ID, Source IDs, Document IDs, Tokenizer hash, Token count, Language
and script, Capability lane, License and provenance tier, Cleaning pipeline hash, Deduplication
status, Contamination status, Evaluation or test overlap status, Content hash, Parent shard IDs.

- Content hash identifies the shard.
- Tokenizer hash gives meaning to the token IDs.
- Cleaning pipeline hash explains how raw text became admitted training data.
- Contamination status determines whether the shard is allowed to enter training.
- Capability lane tells the mixture scheduler where the shard may be used.

Widget: shard manifest builder — assemble a shard from source metadata, tokenizer version,
cleaning pipeline, dedup status, license tier, language tag, contamination scan. Admission gate
blocks shards with unsafe license, missing tokenizer hash, unknown cleaning lineage, or eval
overlap. Live manifest renders as JSON.

Once admitted, a shard behaves like a sealed object — modifying it creates a new shard with a
new hash and new lineage.

## 7. Compiling the Mixture Timeline

Session 5 described the mixture/curriculum in human terms; Session 6 converts it into per-step
quotas. The schedule must know: current stage, tokens belonging to that stage, active
capability lanes, share per lane, protected floors, annealing reserve, transition warmup.

Example stage record:

```json
{
  "stage": "reasoning-heavy-midtrain",
  "token_start": 1800000000000,
  "token_end": 2400000000000,
  "sequence_length": 8192,
  "mixture": {
    "general_web": 0.32,
    "code": 0.22,
    "math_science": 0.18,
    "indic": 0.12,
    "agentic": 0.06,
    "reasoning": 0.10
  },
  "protected_floors": {
    "indic": 0.08,
    "agentic": 0.03,
    "reasoning": 0.05
  },
  "warmup_tokens": 20000000000
}
```

The schedule must also understand data scarcity: if a lane requests more verified tokens than
are available, the plan must specify whether to repeat existing data, generate synthetic data,
reduce the lane share, or move the share to a later stage. If a lane (e.g. agentic Tier A
trajectories) is scarce, the schedule must reserve it rather than letting it exhaust early.

Widget: mixture timeline compiler — set token ranges, mixture weights, protected floors, anneal
reserves, Indic tier splits, reasoning-length bands, warmup widths. Highlights lanes that cannot
be satisfied from available shards; shows where repetition/synthetic/schedule changes are
required.

This is the bridge from Session 5's plan to the actual stream.

## 8. The Training Consumption Ledger

A planned schedule is necessary but the run also needs a record of what actually happened. At
120B scale: workers restart, ranks retry, checkpoints are restored, selection policies change,
files become unavailable. Even with a seeded/indexed planned order, the run needs an append-only
record of the actual consumed stream.

Per consumed batch, the ledger records: Run ID, Branch ID, Global step, Checkpoint ID, Rank,
Microbatch ID, Packed sample IDs, Shard IDs, Token span IDs, Loss mask hash, Attention and
position policy, Mixture lane, Curriculum stage, Tokenizer version, Dataloader version, OPUS
decision ID (where applicable).

This ledger is the run's memory — it lets us reconstruct which tokens contributed to a
checkpoint, resume from a ledger offset, and investigate suspicious model behaviour by locating
the data that shaped a specific training interval.

Widget: training consumption ledger — appends events as microbatches are served. Clicking a
global step reconstructs packed samples, shard ids, token spans, loss-mask hash, mixture lane,
curriculum stage, checkpoint id. A "crash" event shows the next consumed batch being recovered
from the ledger offset.

## 9. Why the Ledger Matters

Scenario: comparing two strategies from an old checkpoint.

- **Without ledger binding**: restore checkpoint, let the dataloader produce whatever the
  current seed/worker-count/shard-set generate. If the new strategy performs better, we can't
  tell if the improvement came from the strategy or from seeing different data.
- **With ledger binding**: restore checkpoint, bind the run to a ledger branch — replay the
  historical data segment, or intentionally fork into a new branch where every data difference
  is explicit. The experiment now has a defined model state AND a defined data stream, making
  the result comparable.

Rule for serious training experiments:

```
experiment = model checkpoint + optimizer state + data stream + code/config
```

If any one of these changes silently, the comparison is weakened.

Widget: checkpoint comparison lab — two branches from the same old checkpoint. One has no ledger
binding and receives a different data stream after resume; the other replays the historical
stream or forks with a new branch id. Loss curves and benchmark deltas annotated to show which
comparison is trustworthy vs. confounded by hidden data differences.

The ledger turns training history into an experiment object.

## 10. OPUS Audit Trail

OPUS sits inside the data path — its decisions are training events. In Session 5, OPUS selected
candidate batches by estimating which updates would be most useful against a proxy direction, so
the data stream contains both accepted batches and scored-and-rejected candidates.

Rejections are valuable — they show what the selector considered low value, what protected
floors rescued, what the model was already comfortable with, and what may deserve review later.

Per candidate batch, the ledger records: Candidate ID, Shard IDs, Capability lane, Curriculum
stage, Model checkpoint used for scoring, Proxy version, OPUS score, Accepted/rejected/deferred
status, Rejection reason, Protected-floor override, Effective-token estimate.

Rejected clean data should not disappear: it may be low value now and valuable later; a rejected
Indic/agentic batch may reveal proxy bias; a rejected code batch may be redundant mid-train but
useful at anneal.

Widget: OPUS accepted/rejected board — candidate batches stream in with lane tags, proxy scores,
shard ids. OPUS accepts/rejects/defers/rescues (protected floor). Keeps four ledgers: accepted,
rejected, deferred, protected. Selecting a rejected batch shows whether it failed due to low
proxy utility, quota pressure, duplication, stage mismatch, or protected-lane bias.

OPUS improves the stream only when its decisions are recorded as part of the stream.

## 11. Token-Level Perplexity Trace

The most detailed learning signal is at the token level. For every loss-bearing token:

```
loss_t = -log p(true token_t)
ppl_t  = exp(loss_t)
```

If this signal is discarded, recovering it later requires re-running the same model over the
same data at the same training state — expensive and possibly unreproducible at scale.

Per-token perplexity reveals patterns shard-level averages hide: an Indic shard may look fine on
average while a specific conjunct/joiner/transliteration stays hard; a code shard may look useful
overall while indentation/rare library calls/error messages carry most of the difficulty; an
agentic trajectory may be easy in the final answer but hard in tool-call arguments; a long
reasoning trace may get easier early while staying hard near the verification step.

Trace fields per loss-bearing token: Token ID, Decoded preview (where allowed), Position in
packed sequence, Document ID, Shard ID, Language and script, Capability lane, Special-token
flag, Boundary/EOS flag, Loss-mask flag, Cross-entropy loss, Token perplexity, Model age when
seen, Checkpoint before/after, Curriculum stage, OPUS score/decision ID, Repeated-pass number.

Storage strategy by level: full token-level traces for proxy/debug runs and suspicious shards;
compressed/quantized token losses for selected intervals of a large run; aggregated statistics
(by shard, language, token ID, position bucket, capability lane, model phase) for the full run.

Widget: token perplexity heatmap over one packed batch — tokens colored by perplexity, filters
for language, shard, position, capability lane, OPUS score, model phase. Switching
early/mid/late checkpoints shows some tokens becoming easy while others stay difficult. A shard
is not uniformly useful or useless — the pattern of surprise inside it is the map for future
data collection. One of the most valuable signals to save for V6.

## 12. The Two-Way Learning Ledger

The consumption ledger records what the model saw; the learning ledger attaches the outcome back
to the data. Per shard/sample/capability-lane/token-cluster: Average token loss, High-perplexity
token clusters, Loss delta before/after exposure, Gradient norm, Gradient alignment (where
available), OPUS score, Repeated-pass effect, Model phase (early/mid/late/anneal), Tokens
consumed when seen, Useful/neutral/harmful classification for future planning.

This creates feedback for the next training run:
- High OPUS score + strong early loss improvement -> likely useful foundation data.
- High OPUS score + little loss reduction -> proxy may be overvaluing it.
- OPUS rejected but perplexity stayed high for that language/pattern -> proxy may be missing a
  scarce capability.
- Repeated passes stop improving loss -> repetition budget exhausted.
- Shard causes gradient spikes -> may need cleaning, staging, or warmup.

The data system becomes a measuring instrument: what was consumed, how surprising it was, how
the model responded, and when during training that response occurred.

Widget: shard learning report card — follows one shard across early/mid/late/anneal phases.
Shows OPUS score, average token perplexity, loss delta, gradient norm, repeated-pass effect,
final usefulness classification. Compare a shard that helped early, one that helped late, one
that caused spikes, and one OPUS undervalued.

This is how V5 teaches V6 what to collect, protect, repeat, defer, or reject.

## 13. Test Shards and the Eval Firewall

Evaluation data also needs manifests — the difference is permission. Training shards are
admitted into the data stream; **test shards are registered in the audit system and blocked
from training**.

Test shard fields: Content hashes, Benchmark IDs, Version tags, Contamination fingerprints,
Access logs, Never-train flag.

The system must know evaluation data exists precisely so it can prevent it from entering a
training batch. This firewall was introduced in Session 3, reinforced in Session 4, and made
**executable** in Session 6.

- When a candidate shard is selected, the dataloader checks the evaluation registry.
- When a benchmark is updated, its hashes are added to the registry.
- When a suspicious score increase appears, the run can be audited against test-shard
  fingerprints and canary strings.

Same discipline applies to validation shards — they may be read during training for evaluation,
but must never become gradient-bearing training data.

Widget: eval firewall — train/validation/test shards in one registry with different permissions.
A candidate training batch is checked against content hashes, contamination fingerprints,
benchmark ids. Injecting an overlap blocks the batch and writes a rejection event. A separate
audit view traces a later benchmark jump back to the firewall.

The model can only be trusted when the test data has a memory too.

## 14. Resume, Replay, and Fork

Large training runs are interrupted; the data system must define exactly what happens next.
Four modes:

- **Resume**: continue the same run from the latest checkpoint and latest ledger offset. Normal
  crash-recovery path.
- **Replay**: restore an older checkpoint and feed the same historical data stream from that
  point. Useful for comparing a code/training change under identical data exposure.
- **Fork**: restore a checkpoint and intentionally start a new data branch. The new branch gets
  a new ID; the ledger records the exact divergence point.
- **Audit**: reconstruct the data that trained a checkpoint or range of checkpoints. Answers
  questions like "which shards influenced the model between 5.4B and 5.6B tokens?" or "which
  OPUS-selected batches appeared before a loss spike?"

Checkpointing binds model state to data state — **a checkpoint without a data position is
incomplete.**

Widget: crash, replay and fork drill — training advances through ledger offsets, crashes,
resumes, rolls back, forks. Shows checkpoint step, ledger offset, branch id, next batch. Choose
replay or fork after restoring an old checkpoint; the ledger keeps the experiment definition
explicit.

This is the operational heart of the session: **data state travels with model state.**

## 15. Dataloader Throughput

Correctness alone does not keep GPUs busy. The dataloader must deliver useful tokens faster than
GPUs consume them. Throughput depends on: Shard size, Compression, Storage bandwidth, Local
caching, Prefetch depth, Worker count, Rank partitioning, Packing efficiency, OPUS rejection
rate.

Small files create overhead; very large files reduce flexibility and complicate recovery.
Compression saves storage/network bandwidth but consumes CPU. Prefetching hides latency but uses
memory. More workers help until they saturate storage or compete with each other. OPUS may
improve token value while reducing accepted-token throughput if candidate generation/scoring is
slow.

The metric that matters: **useful loss-bearing tokens per second at the target mixture.** A
loader may report high raw token throughput while wasting compute on padding, context-only
tokens, or OPUS-rejected batches.

Metrics to monitor: Raw tokens/sec, Useful loss-bearing tokens/sec, Accepted tokens/sec after
OPUS, GPU idle time, Loader wait time, Cache hit rate, Shard read latency, Packing utilization,
Rejection rate by lane, Replay and resume latency.

Widget: dataloader throughput lab — adjust shard size, compression, worker count, prefetch
depth, cache hit rate, packing efficiency, OPUS rejection rate. Output shows raw tokens/sec,
useful tokens/sec, accepted tokens/sec, GPU idle time, loader wait time. Key distinction: a fast
loader can still deliver little learning when much of the stream is padding, context-only
tokens, or rejected candidates.

The best data system is both inspectable and fast enough to disappear from the GPU's point of
view.

## 16. The Assignment

(See `S6-assignment.md` for the full, verbatim assignment text and grading rubric — this section
of the lesson page is duplicated there.)

## 17. References

- Megatron Core — indexed GPT datasets through document, sample and shuffle indices (baseline
  mental model for planned sample lookup).
- Mosaic StreamingDataset — mid-epoch resumable streaming and deterministic sample ordering.
- NVIDIA NeMo Curator — large-scale curation, exact/fuzzy/semantic deduplication, GPU-accelerated
  data processing.
- WebDataset-style sharded tar formats — useful for structured examples where multiple files
  belong to one sample.
- LakeFS, Iceberg, Delta-style transaction logs — database/lakehouse analogies for versioned
  data, replay and audit history.
