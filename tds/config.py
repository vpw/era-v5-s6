"""Run configuration loading and path resolution.

The whole demo is driven by `config/run_config.json`. Its hash travels into the evidence
bundle so a set of artifacts can be tied back to the configuration that produced it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hashing import sha256_file

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "run_config.json"


class Config:
    """Thin wrapper over the config dict: dotted lookup plus path resolution."""

    def __init__(self, data: dict, source: Path):
        self.data = data
        self.source = Path(source)
        self.config_hash = sha256_file(self.source)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG
        with open(path, encoding="utf-8") as fh:
            return cls(json.load(fh), path)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise KeyError(f"missing required config key: {dotted}")
        return value

    # -- paths ------------------------------------------------------------------

    def path(self, dotted: str) -> Path:
        """Resolve a configured path relative to the repository root."""
        return REPO_ROOT / str(self.require(dotted))

    @property
    def work_dir(self) -> Path:
        return self.path("paths.work_dir")

    @property
    def shard_dir(self) -> Path:
        return self.path("paths.shard_dir")

    @property
    def artifacts_dir(self) -> Path:
        return self.path("paths.artifacts_dir")

    @property
    def lanes(self) -> list[str]:
        """Lane names in a fixed order, so iteration never depends on dict ordering."""
        return sorted(k for k in self.require("lanes") if not k.startswith("_"))

    def lane(self, name: str) -> dict:
        return self.require("lanes")[name]

    def stage_for_token(self, token_position: int) -> str:
        """Which curriculum stage a given absolute token position falls in."""
        stages = self.require("mixture.stages")
        for stage in stages:
            if stage["token_start"] <= token_position < stage["token_end"]:
                return stage["stage"]
        return stages[-1]["stage"]

    def model_age_bucket(self, tokens_seen: int) -> str:
        """OPUS conditions on model age; widget 10 exposes this as a dropdown."""
        for bucket in self.require("opus.model_age_buckets"):
            if tokens_seen < bucket["until_tokens"]:
                return bucket["name"]
        return self.require("opus.model_age_buckets")[-1]["name"]

    def fingerprint(self) -> dict:
        """Config identity, recorded in the evidence bundle."""
        return {
            "config_path": str(self.source.relative_to(REPO_ROOT)),
            "config_sha256": self.config_hash,
            "run_id": self.require("run.run_id"),
            "seed": self.require("run.seed"),
        }
