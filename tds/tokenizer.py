"""The frozen tokenizer.

Session 2's contract, restated by widget 1: a training shard is only meaningful under the
exact tokenizer that created it, including special tokens and normalisation choices. So
this module does three things and no more -- load one tokenizer file, hash it, and encode
documents under it. There is no path here that trains, extends or mutates a vocabulary.

The hash is over the tokenizer file's bytes, which covers the vocab, the merge table, the
added-token table and the normaliser configuration in one value. Every shard manifest
records it, and the audit recomputes it from the file rather than trusting the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer

from .config import REPO_ROOT, Config
from .corpus import Document, Segment
from .hashing import sha256_file, tagged


@dataclass(frozen=True)
class EncodedSegment:
    """A document segment after tokenization, with its role preserved.

    Roles survive tokenization because the loss mask is decided per role, not per
    document: in an agentic trajectory the user turn and the tool observation are
    context, and only the assistant's own tokens are scored.
    """

    role: str
    ids: tuple[int, ...]


class FrozenTokenizer:
    def __init__(self, config: Config):
        self.path = REPO_ROOT / config.require("tokenizer.path")
        if not self.path.exists():
            raise FileNotFoundError(
                f"tokenizer not found at {self.path}; run scripts/build_fixtures.py"
            )
        self._tok = HFTokenizer.from_file(str(self.path))
        self.name = config.require("tokenizer.name")
        self.normalizer_id = config.require("tokenizer.normalizer_id")

        specials = config.require("tokenizer.special_tokens")
        self.special_tokens = dict(specials)
        self.bos_id = self._require_token(specials["bos"])
        self.eos_id = self._require_token(specials["eos"])
        self.unk_id = self._require_token(specials["unk"])
        self.pad_id = self._require_token(specials["pad"])

        self.file_sha256 = sha256_file(self.path)
        self.tokenizer_hash = tagged("tok", self.file_sha256)
        self.vocab_size = self._tok.get_vocab_size()

    def _require_token(self, token: str) -> int:
        token_id = self._tok.token_to_id(token)
        if token_id is None:
            raise ValueError(
                f"special token {token!r} is absent from {self.path.name}; "
                "the configured special-token policy does not match the frozen vocab"
            )
        return token_id

    # -- encoding ----------------------------------------------------------------

    def encode_text(self, text: str) -> list[int]:
        """Encode without any implicit special tokens.

        The tokenizer file carries a post-processor that would prepend `<s>`. Boundary
        tokens are this system's business, not the tokenizer's -- the packer decides
        where EOS goes -- so they are suppressed here and inserted deliberately.
        """
        return self._tok.encode(text, add_special_tokens=False).ids

    def encode_document(self, doc: Document) -> list[EncodedSegment]:
        return [
            EncodedSegment(role=segment.role, ids=tuple(self.encode_text(segment.text)))
            for segment in doc.segments
        ]

    def decode(self, ids) -> str:
        return self._tok.decode(list(ids))

    # -- identity ----------------------------------------------------------------

    def descriptor(self) -> dict:
        """What every shard manifest records about the tokenizer that built it."""
        return {
            "tokenizer_name": self.name,
            "tokenizer_hash": self.tokenizer_hash,
            "tokenizer_sha256": self.file_sha256,
            "tokenizer_file": str(self.path.relative_to(REPO_ROOT)),
            "vocab_size": self.vocab_size,
            "normalizer_id": self.normalizer_id,
            "special_tokens": {
                name: {"token": token, "id": self._tok.token_to_id(token)}
                for name, token in sorted(self.special_tokens.items())
            },
        }

    def verify(self, expected_hash: str) -> bool:
        """Recompute the hash from the file on disk and compare."""
        return tagged("tok", sha256_file(self.path)) == expected_hash
