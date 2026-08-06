"""Checkpoints, bound to a ledger offset.

The session's rule is that an experiment is `model checkpoint + optimizer state + data
stream + code/config`, and widget 9 dramatises what happens when you keep the first two
and let the third drift: two branches from the same checkpoint diverge, and the loss delta
cannot be attributed to the change you were testing.

So a checkpoint here saves four things, not two:

    model_state       parameters
    optimizer_state   Adam moments and step count
    dataloader_state  the ledger offset and global step the data stream had reached
    config/code       the config hash and the shard registry hash it was trained under

`next_step` is what makes resume exact. It is the step the run had *not yet* committed, so
restoring and continuing consumes precisely the batch that was about to be consumed --
no skip, no repeat.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .hashing import sha256_file, sha256_json, tagged


@dataclass
class CheckpointMeta:
    checkpoint_id: str
    global_step: int
    next_step: int
    ledger_offset: int
    branch_id: str
    model_hash: str
    config_sha256: str
    registry_hash: str
    tokens_consumed: int
    parent_checkpoint_id: str | None = None
    parent_branch_id: str | None = None

    def to_record(self) -> dict:
        return {
            "checkpoint_id": self.checkpoint_id,
            "global_step": self.global_step,
            "next_step": self.next_step,
            "ledger_offset": self.ledger_offset,
            "dataloader_state": f"ledger_offset_{self.ledger_offset}",
            "branch_id": self.branch_id,
            "model_hash": self.model_hash,
            "config_sha256": self.config_sha256,
            "registry_hash": self.registry_hash,
            "tokens_consumed": self.tokens_consumed,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "parent_branch_id": self.parent_branch_id,
        }


class CheckpointStore:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, checkpoint_id: str) -> Path:
        return self.directory / f"{checkpoint_id}.npz"

    def meta_path_for(self, checkpoint_id: str) -> Path:
        return self.directory / f"{checkpoint_id}.meta.json"

    def save(self, checkpoint_id: str, model, optimizer, meta: CheckpointMeta) -> Path:
        payload = {f"param::{name}": value for name, value in model.params.items()}
        payload.update({f"opt::{name}": value for name, value in optimizer.state().items()})
        path = self.path_for(checkpoint_id)
        np.savez(path, **payload)

        record = meta.to_record()
        record["state_sha256"] = sha256_file(path)
        with open(self.meta_path_for(checkpoint_id), "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=1, sort_keys=True)
        return path

    def load(self, checkpoint_id: str, model, optimizer) -> dict:
        path = self.path_for(checkpoint_id)
        with np.load(path) as archive:
            for key in archive.files:
                if key.startswith("param::"):
                    model.params[key[7:]] = archive[key].astype(model.config.dtype)
            optimizer.load_state(
                {key[5:]: archive[key] for key in archive.files if key.startswith("opt::")}
            )
        with open(self.meta_path_for(checkpoint_id), encoding="utf-8") as fh:
            record = json.load(fh)

        # A checkpoint that does not restore the parameters it saved is worse than no
        # checkpoint, because everything downstream would still look consistent.
        if model.state_hash() != record["model_hash"]:
            raise RuntimeError(
                f"checkpoint {checkpoint_id} restored to model hash {model.state_hash()}, "
                f"but was saved as {record['model_hash']}"
            )
        return record

    def read_meta(self, checkpoint_id: str) -> dict:
        with open(self.meta_path_for(checkpoint_id), encoding="utf-8") as fh:
            return json.load(fh)

    def list_checkpoints(self) -> list[dict]:
        metas = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.directory.glob("*.meta.json"))
        ]
        return sorted(metas, key=lambda m: (m["branch_id"], m["global_step"]))

    def latest_for_branch(self, branch_id: str) -> dict | None:
        candidates = [m for m in self.list_checkpoints() if m["branch_id"] == branch_id]
        return candidates[-1] if candidates else None

    def write_index(self) -> Path:
        checkpoints = self.list_checkpoints()
        index = {
            "checkpoints": len(checkpoints),
            "by_branch": {},
            "entries": checkpoints,
        }
        for meta in checkpoints:
            index["by_branch"].setdefault(meta["branch_id"], []).append(meta["checkpoint_id"])
        path = self.directory / "checkpoint_index.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(index, fh, ensure_ascii=False, indent=1, sort_keys=True)
        return path


def checkpoint_id_for(step: int) -> str:
    """Widget 8's naming: ckpt_00004."""
    return f"ckpt_{step:05d}"


def registry_hash(registry) -> str:
    """Identity of the data the run is allowed to see.

    Part of the checkpoint because restoring a checkpoint against a different admitted
    shard set is a different experiment, even if the model weights match.
    """
    return tagged(
        "registry",
        sha256_json(
            [
                {"shard_id": e.shard_id, "content_hash": e.manifest["content_hash"]}
                for e in registry.admitted
            ]
        ),
    )
