# token masking, argmax, choice among strings
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.vocab import Vocab

Mask = np.ndarray


def mask_logits(logits: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Return a copy of `logits` with all non-allowed indices set to -inf."""
    if not allowed.any():
        raise ValueError("empty allowed mask: no legal next token")
    masked = np.full_like(logits, -np.inf)
    masked[allowed] = logits[allowed]
    return masked


class SharedMasks(BaseModel):
    """Vocab-sized boolean masks reused by every value constraint"""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True
    )

    size: int

    # Primitive character masks
    digits: Mask            # '0'..'9'
    minus: Mask             # '-'
    dot: Mask               # '.'
    comma: Mask             # ','
    close_brace: Mask       # '}'
    quote: Mask             # '"'

    # Pre-combined number masks (avoids | at every step)
    number_start: Mask      # minus | digits
    number_continue: Mask   # dot   | digits

    # String content: decodable token, no '"', no '\\', no ctrl bytes
    string_content: Mask

    @classmethod
    def build(cls, vocab: Vocab, size: int) -> SharedMasks:
        """Build every mask from the vocabulary"""
        digits = {
            tid for tid, raw in vocab.items()
            if all(0x30 <= b <= 0x39 for b in raw)
        }
        minus = vocab.bytes_to_ids[b"-"]
        dot = vocab.bytes_to_ids[b"."]
        comma = vocab.bytes_to_ids[b","]
        close_brace = vocab.bytes_to_ids[b"}"]
        quote = vocab.bytes_to_ids[b'"']

        string_content_ids: set[int] = set()
        for tid, _raw in vocab.items():
            text = vocab.token_text(tid)
            if text is None:
                continue                         # split-byte token
            if '"' in text or "\\" in text:
                continue
            if any(ord(c) < 0x20 for c in text):
                continue
            string_content_ids.add(tid)

        return cls(
            size=size,
            digits=_make_mask(size, digits),
            minus=_make_mask(size, {minus}),
            dot=_make_mask(size, {dot}),
            comma=_make_mask(size, {comma}),
            close_brace=_make_mask(size, {close_brace}),
            quote=_make_mask(size, {quote}),
            number_start=_make_mask(size, digits | {minus}),
            number_continue=_make_mask(size, digits | {dot}),
            string_content=_make_mask(size, string_content_ids),
        )


def _make_mask(size: int, ids: Iterable[int]) -> np.ndarray:
    mask: Mask = np.zeros(size, dtype=bool)
    mask[list(ids)] = True
    return mask


@dataclass(frozen=True)
class ChoiceConstraint:
    """Finite choices: function names, booleans"""

    options: tuple[bytes, ...]
    vocab: Vocab
    masks: SharedMasks

    def allowed(self, buf: bytes) -> np.ndarray:
        mask: Mask = np.zeros(self.masks.size, dtype=bool)
        for opt in self.options:
            if not opt.startswith(buf):
                continue
            rest = opt[len(buf):]
            for k in range(1, len(rest) + 1):
                tid = self.vocab.bytes_to_ids.get(rest[:k])
                if tid is not None and tid < self.masks.size:
                    mask[tid] = True
        return mask

    def is_complete(self, buf: bytes) -> bool:
        return buf in self.options

    def value(self, buf: bytes) -> str:
        if not self.is_complete(buf):
            raise ValueError(f"incomplete choice: {buf!r}")
        text = buf.decode("utf-8")
        return text[:-1] if text.endswith('"') else text


@dataclass(frozen=True)
class NumberConstraint:
    """JSON number: ``-?digits(.digits)?`` followed by a terminator."""

    masks: SharedMasks
    terminator: Mask
    as_type: Literal["int", "float"] = "float"

    def allowed(self, buf: bytes) -> Mask:
        state = _number_state(buf)
        if state == "start":
            return self.masks.number_start
        if state in ("after_minus", "after_dot"):
            return self.masks.digits
        if state == "int":
            if self.as_type == "int":
                res_int: Mask = self.masks.digits | self.terminator
                return res_int
            res: Mask = self.masks.number_continue | self.terminator
            return res
        if state == "frac":
            res_frac: Mask = self.masks.digits | self.terminator
            return res_frac
        empty: Mask = np.zeros(self.masks.size, dtype=bool)
        return empty

    def is_complete(self, buf: bytes) -> bool:
        if not buf or not (buf.endswith(b",") or buf.endswith(b"}")):
            return False
        core = buf[:-1]
        return _number_state(core) in ("int", "frac")

    def value(self, buf: bytes) -> int | float:
        if not self.is_complete(buf):
            raise ValueError(f"incomplete number: {buf!r}")
        body = buf[:-1]
        if self.as_type == "int":
            return int(body)
        return float(body)


@dataclass(frozen=True)
class StringConstraint:
    """Free-form string value"""

    masks: SharedMasks

    def allowed(self, buf: bytes) -> np.ndarray:
        if buf.endswith(b'"'):
            return np.zeros(self.masks.size, dtype=bool)
        res: np.ndarray = self.masks.string_content | self.masks.quote
        return res

    def is_complete(self, buf: bytes) -> bool:
        return buf.endswith(b'"')

    def value(self, buf: bytes) -> str:
        if not self.is_complete(buf):
            raise ValueError(f"incomplete string: {buf!r}")
        return buf[:-1].decode("utf-8")


def _number_state(buf: bytes) -> str:
    """Classify a number buffer (terminator included) into a state.

    States:
        start       : empty
        after_minus : b"-"
        int         : one or more digits, no '.'
        after_dot   : b"<int>."
        frac        : b"<int>.<digits>"  (complete)
    A trailing ',' or '}' is stripped before classification.
    """
    if buf and buf[-1:] in (b",", b"}"):
        buf = buf[:-1]
    if not buf:
        return "start"
    if buf == b"-":
        return "after_minus"
    sign = buf[1:] if buf.startswith(b"-") else buf
    if not sign:
        return "after_minus"
    if b"." not in sign:
        return "int"
    _, frac = sign.split(b".", 1)
    return "after_dot" if not frac else "frac"
