"""Shared fixtures.

Most tests here are unit-level and build their own tiny inputs, so the suite stays fast
and does not depend on a demo run having happened. The handful of tests that check the
*generated artifacts* are skipped with a clear message when `submission_artifacts/` is
absent, rather than failing -- a fresh clone should be able to run `pytest` before
`run_demo.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tds.config import Config  # noqa: E402


@pytest.fixture(scope="session")
def config() -> Config:
    return Config.load()


@pytest.fixture(scope="session")
def artifacts(config):
    root = config.artifacts_dir
    if not (root / "evidence.json").exists():
        pytest.skip("no submission_artifacts/ yet -- run `python run_demo.py` first")
    return root


@pytest.fixture(scope="session")
def shard_dir(config):
    directory = config.shard_dir
    if not directory.exists() or not any(directory.glob("*.tokens.bin")):
        pytest.skip("no shards yet -- run `python run_demo.py` first")
    return directory
