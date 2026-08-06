"""Training Data Execution System (ERA V5, Session 6).

The path this package implements, end to end:

    documents -> tokenized shards -> manifests -> mixture schedule -> packing
    -> batches -> training -> consumption ledger -> learning ledger
    -> checkpoint -> crash -> resume -> replay -> audit

The organising idea is that the batch stream is a *pure function* of
(seed, branch_id, step, schedule, admitted shard registry) and the ledger records what
that function produced. Resume, replay and fork are then the same mechanism viewed
three ways, rather than three separate features.
"""

__version__ = "1.0.0"
