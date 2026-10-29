"""Tests for src.constraint: which tokens may come next.

No model needed: everything runs on a tiny fake vocab.
"""
import re
from typing import Any

import numpy as np
import pytest

from src.constraint import (
    ChoiceConstraint,
    NumberConstraint,
    SharedMasks,
    StringConstraint,
    mask_logits,
)
from src.vocab import Vocab

TOKENS: list[bytes] = [
    *(str(d).encode() for d in range(10)),               # 0-9
    b"-", b".", b",", b"}", b'"',                        # 10-14
    b"12", b"1.5", b".5", b"-.", b"12,", b'",', b"},",   # 15-21
    b"fn", b"fn_", b"fn_add", b"fn_add_numbers",         # 22-25
    b"_", b"_numbers", b"add",                           # 26-28
    b"x", b"hello", b'llo"', b'a"b', b"\\",              # 29-33
    b"\n", b"\t", b" ",                                  # 34-36
    "é".encode(), b"\xc3", "😀".encode(),                # 37-39
    b"true", b"false", b"t", b"tr", b"r", b"ue",         # 40-45
    b"fa", b"lse",                                       # 46-47
    *(c.encode() for c in "fnadumbesl"),                 # 48-57
]
SIZE = len(TOKENS) + 8       # logits are wider than the vocab (padding ids)
DIGITS = [str(d).encode() for d in range(10)]


@pytest.fixture(scope="module")
def vocab() -> Vocab:
    return Vocab(id_to_bytes=dict(enumerate(TOKENS)))


@pytest.fixture(scope="module")
def masks(vocab: Vocab) -> SharedMasks:
    return SharedMasks.build(vocab, SIZE)


def toks(mask: np.ndarray) -> set[bytes]:
    """Readable view of a mask: the set of allowed token byte strings."""
    assert mask.dtype == bool and mask.shape == (SIZE,)
    idx = np.flatnonzero(mask)
    assert all(i < len(TOKENS) for i in idx), "padding id marked allowed"
    return {TOKENS[i] for i in idx}


def check(mask: np.ndarray, yes: list[bytes], no: list[bytes]) -> None:
    got = toks(mask)
    for t in yes:
        assert t in got, f"{t!r} should be allowed"
    for t in no:
        assert t not in got, f"{t!r} should NOT be allowed"


# --------------------------------------------------------------------------
# mask_logits
# --------------------------------------------------------------------------

def test_mask_logits_keeps_allowed_and_blocks_the_rest() -> None:
    logits = np.array([1.0, 5.0, 2.0, 3.0])
    out = mask_logits(logits, np.array([True, False, True, False]))
    assert out[0] == 1.0 and out[2] == 2.0
    assert np.isneginf(out[1]) and np.isneginf(out[3])


def test_mask_logits_does_not_mutate_input() -> None:
    logits = np.array([1.0, 5.0, 2.0])
    mask_logits(logits, np.array([True, False, True]))
    assert logits.tolist() == [1.0, 5.0, 2.0]


def test_argmax_after_masking_is_always_allowed() -> None:
    rng = np.random.default_rng(40)
    for _ in range(50):
        logits = rng.standard_normal(30)
        allowed = rng.random(30) < 0.2
        if not allowed.any():
            continue
        best = int(np.argmax(mask_logits(logits, allowed)))
        assert allowed[best]


def test_mask_logits_empty_mask_raises() -> None:
    with pytest.raises(ValueError):
        mask_logits(np.zeros(4), np.zeros(4, dtype=bool))


# --------------------------------------------------------------------------
# ChoiceConstraint: function names and booleans
# --------------------------------------------------------------------------

@pytest.fixture
def names(vocab: Vocab, masks: SharedMasks) -> ChoiceConstraint:
    return ChoiceConstraint(
        options=(b'fn_add"', b'fn_add_numbers"'), vocab=vocab, masks=masks,
    )


@pytest.fixture
def booleans(vocab: Vocab, masks: SharedMasks) -> ChoiceConstraint:
    return ChoiceConstraint(
        options=(b"true", b"false"), vocab=vocab, masks=masks,
    )


def test_choice_first_token(names: ChoiceConstraint) -> None:
    assert toks(names.allowed(b"")) == {
        b"f", b"fn", b"fn_", b"fn_add", b"fn_add_numbers",
    }


def test_choice_follows_the_prefix(names: ChoiceConstraint) -> None:
    assert toks(names.allowed(b"fn")) == {b"_"}
    assert toks(names.allowed(b"fn_add_numbers")) == {b'"'}


def test_choice_can_end_or_continue_when_one_name_prefixes_another(
    names: ChoiceConstraint,
) -> None:
    assert toks(names.allowed(b"fn_add")) == {b'"', b"_", b"_numbers"}


def test_choice_never_crosses_the_terminator(names: ChoiceConstraint) -> None:
    assert b'",' not in toks(names.allowed(b"fn_add"))


def test_choice_completion_needs_the_terminator(
    names: ChoiceConstraint,
) -> None:
    assert not names.is_complete(b"")
    assert not names.is_complete(b"fn_add")
    assert names.is_complete(b'fn_add"')
    assert names.is_complete(b'fn_add_numbers"')


def test_choice_value_strips_the_terminator(names: ChoiceConstraint) -> None:
    assert names.value(b'fn_add"') == "fn_add"
    with pytest.raises(ValueError):
        names.value(b"fn_add")


def test_boolean_choice(booleans: ChoiceConstraint) -> None:
    assert toks(booleans.allowed(b"")) == {
        b"true", b"false", b"t", b"tr", b"f", b"fa",
    }
    assert toks(booleans.allowed(b"t")) == {b"r"}
    assert toks(booleans.allowed(b"tr")) == {b"u", b"ue"}
    assert toks(booleans.allowed(b"fa")) == {b"l", b"lse"}
    assert booleans.is_complete(b"true")
    assert not booleans.is_complete(b"tr")
    assert booleans.value(b"false") == "false"


def test_choice_buffer_that_is_not_a_prefix_allows_nothing(
    names: ChoiceConstraint,
) -> None:
    assert not names.allowed(b"nope").any()


def test_choice_padding_ids_are_never_allowed(names: ChoiceConstraint) -> None:
    assert not names.allowed(b"")[len(TOKENS):].any()


# --------------------------------------------------------------------------
# NumberConstraint
# --------------------------------------------------------------------------

def num(masks: SharedMasks, term: str = "comma", **kw: Any) -> NumberConstraint:
    t = masks.comma if term == "comma" else masks.close_brace
    return NumberConstraint(masks=masks, terminator=t, **kw)


@pytest.mark.parametrize(
    "buf, yes, no",
    [
        (b"", [*DIGITS, b"-", b"12"],
            [b".", b",", b"}", b"x", b" ", b"-."]),
        (b"-", [*DIGITS, b"12"], [b"-", b".", b",", b"}"]),
        (b"0", [*DIGITS, b".", b","], [b"-"]),
        (b"12", [*DIGITS, b"12", b".", b","], [b"-", b"12,", b"},"]),
        (b"12.", [*DIGITS, b"12"], [b".", b",", b"}", b"-"]),
    ],
)
def test_number_grammar(
    masks: SharedMasks, buf: bytes, yes: list[bytes], no: list[bytes],
) -> None:
    check(num(masks).allowed(buf), yes, no)


def test_number_mixed_tokens_are_never_allowed(masks: SharedMasks) -> None:
    for buf in (b"", b"1", b"12."):
        check(num(masks).allowed(buf), [], [b"1.5", b".5", b"-.", b"12,"])


def test_integer_never_allows_a_dot(masks: SharedMasks) -> None:
    c = num(masks, as_type="int")
    check(c.allowed(b""), [*DIGITS, b"-"], [b".", b"1.5"])
    check(c.allowed(b"0"), [*DIGITS, b","], [b"."])
    check(c.allowed(b"12"), [*DIGITS, b","], [b"."])


def test_number_terminator_is_exact(masks: SharedMasks) -> None:
    check(num(masks, "comma").allowed(b"12"), [b","], [b"}", b"12,", b"},"])
    check(num(masks, "brace").allowed(b"12"), [b"}"], [b",", b"12,", b"},"])


@pytest.mark.parametrize("buf", [b"", b"-", b"12.", b"-0."])
def test_number_cannot_end_while_incomplete(
    masks: SharedMasks, buf: bytes,
) -> None:
    t = toks(num(masks).allowed(buf))
    assert b"," not in t and b"}" not in t


def test_number_completion(masks: SharedMasks) -> None:
    c = num(masks)
    assert c.is_complete(b"12,")
    assert c.is_complete(b"-0.5,")
    assert num(masks, "brace").is_complete(b"7.5}")
    assert not c.is_complete(b"12")
    assert not c.is_complete(b",")
    assert not c.is_complete(b"-,")
    assert not c.is_complete(b"12.,")
    assert not c.is_complete(b"")


def test_number_value_float(masks: SharedMasks) -> None:
    c = num(masks)
    assert c.value(b"-0.5,") == -0.5
    v = c.value(b"2,")
    assert v == 2.0 and isinstance(v, float)


def test_number_value_int(masks: SharedMasks) -> None:
    c = num(masks, as_type="int")
    v = c.value(b"-42,")
    assert v == -42 and isinstance(v, int)


def test_number_value_requires_completeness(masks: SharedMasks) -> None:
    with pytest.raises(ValueError):
        num(masks).value(b"12")


# --------------------------------------------------------------------------
# StringConstraint
# --------------------------------------------------------------------------

@pytest.fixture
def strings(masks: SharedMasks) -> StringConstraint:
    return StringConstraint(masks=masks)


def test_string_allows_safe_content_and_closing(
    strings: StringConstraint,
) -> None:
    yes = [b"x", b"hello", b" ", b",", b"}", b"12,", b"-.", *DIGITS,
        "é".encode(), "😀".encode(), b'"']
    check(strings.allowed(b"hello"), yes, [])


def test_string_blocks_unsafe_tokens(strings: StringConstraint) -> None:
    no = [
        b'a"b',
        b'",',
        b'llo"',
        b"\\",
        b"\n", b"\t",
    ]
    check(strings.allowed(b"hello"), [], no)


def test_string_content_is_stateless(strings: StringConstraint) -> None:
    assert (strings.allowed(b"") == strings.allowed(b"hello")).all()


def test_string_nothing_is_allowed_after_the_closing_quote(
    strings: StringConstraint,
) -> None:
    assert not strings.allowed(b'hello"').any()


def test_string_completion_and_value(strings: StringConstraint) -> None:
    assert strings.is_complete(b'hello"')
    assert strings.is_complete(b'"')
    assert not strings.is_complete(b"hello")
    assert not strings.is_complete(b"")
    assert strings.value(b'hello"') == "hello"
    assert strings.value(b'"') == ""
    assert strings.value('café"'.encode()) == "café"
    with pytest.raises(ValueError):
        strings.value(b"hello")


def test_string_padding_ids_are_never_allowed(
    strings: StringConstraint,
) -> None:
    assert not strings.allowed(b"")[len(TOKENS):].any()


# --------------------------------------------------------------------------
# Random walks
# --------------------------------------------------------------------------

def walk(c: Any, seed: int, limit: int = 400) -> bytes:
    rng = np.random.default_rng(seed)
    buf = b""
    for _ in range(limit):
        options = np.flatnonzero(c.allowed(buf))
        assert options.size, f"dead end at {buf!r}"
        buf += TOKENS[int(rng.choice(options))]
        if c.is_complete(buf):
            return buf
    raise AssertionError(f"never terminated: {buf!r}")


@pytest.mark.parametrize("seed", range(40))
def test_walk_number(masks: SharedMasks, seed: int) -> None:
    c = num(masks)
    buf = walk(c, seed)
    text = str(buf[:-1], "ascii")
    assert re.fullmatch(r"-?\d+(\.\d+)?", text), text
    assert isinstance(c.value(buf), float)


@pytest.mark.parametrize("seed", range(40))
def test_walk_integer(masks: SharedMasks, seed: int) -> None:
    c = num(masks, "brace", as_type="int")
    buf = walk(c, seed)
    assert buf.endswith(b"}")
    assert re.fullmatch(r"-?\d+", buf[:-1].decode()), buf
    assert isinstance(c.value(buf), int)


@pytest.mark.parametrize("seed", range(40))
def test_walk_string(strings: StringConstraint, seed: int) -> None:
    buf = walk(strings, seed)
    text = strings.value(buf)
    assert '"' not in text and "\\" not in text
    assert all(ord(ch) >= 0x20 for ch in text)


@pytest.mark.parametrize("seed", range(20))
def test_walk_choice(names: ChoiceConstraint, seed: int) -> None:
    assert names.value(walk(names, seed)) in {"fn_add", "fn_add_numbers"}


@pytest.mark.parametrize("seed", range(20))
def test_walk_boolean(booleans: ChoiceConstraint, seed: int) -> None:
    assert booleans.value(walk(booleans, seed)) in {"true", "false"}