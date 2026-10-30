"""Tests for src.decoder: the loop that turns one request into one call.

No real model: a scripted fake plays the LLM. Three kinds of "model":

  * teacher  - wants one exact output and always prefers the longest token
               that continues it. If the decoder is right, the output is
               exactly that text, whatever the vocab looks like.
  * random   - random logits, and padding ids (ids >= len(vocab)) get a
               huge score. Constraints must still give a valid call.
  * stubborn - always prefers one token (e.g. "x" or "7") and never wants
               to stop. Only the length cap can end the value.

Contract these tests assume (rename in one place if yours differs):

    Decoder(model=, vocab=, functions=list[FunctionDef],
            max_value_tokens=int)       keyword-only
    Decoder.decode(prompt: str) -> OutputResult
    Decoder.last_ids: list[int]         token ids of the model's final
                                        context (prefix + whole call text)
    DecoderError(Exception)             bad function set at construction,
                                        or an impossible state mid-run

    model.encode(text)                  -> 2-D array/tensor, ids in [0]
    model.get_logits_from_input_ids(ids) -> list[float], len >= len(vocab)

What the decoder must guarantee:
  * The context the model ends up with is EXACTLY
        build_prefix(functions) + call_text(prompt, name, params)
    (for numbers the model writes the same digits json.dumps would).
  * Output is built from typed values, never parsed from generated text.
  * Value caps: after max_value_tokens tokens inside a string or a number,
    the closing token is forced when it is legal. A bare "-" or "12."
    can't end yet, so the value is completed with digits first.
  * Literal scaffold text costs no forward pass.
  * Padding ids are never chosen.
  * decode() leaves no state behind.
"""
import math
from typing import Any, Callable

import numpy as np
import pytest

from src.context import build_prefix, call_text, head
from src.decoder import Decoder, DecoderError
from src.models import FunctionDef, OutputResult
from src.vocab import Vocab

# --------------------------------------------------------------------------
# Fake vocab: every single byte (like real byte-level BPE) plus some
# multi-byte tokens, including ones that must NOT be usable in values.
# --------------------------------------------------------------------------

EXTRA: list[bytes] = [
	b'",', b'"}', b"12", b"2024", b"fn", b"fn_", b"_numbers", b"add",
	b"now", b"hello", b" the", b"shrek", b".0", b"0.", b", ", b"\n\n",
	b'a"b', b"true", b"false",
	"é".encode(), "😀".encode(),
]
TOKENS: list[bytes] = [bytes([i]) for i in range(256)] + EXTRA
SIZE = len(TOKENS) + 8                       # logits wider than the vocab
PAD = list(range(len(TOKENS), SIZE))
VOCAB = Vocab(id_to_bytes=dict(enumerate(TOKENS)))


def detok(ids: list[int]) -> bytes:
	return b"".join(TOKENS[i] for i in ids)


class FakeModel:
	"""Greedy longest-match tokenizer + a scorer that plays the LLM."""

	def __init__(self, scorer: Callable[[bytes], list[float]]) -> None:
		self.scorer = scorer
		self.calls: list[list[int]] = []
		self._by_len = sorted(
			((b, i) for i, b in enumerate(TOKENS)), key=lambda t: -len(t[0]))

	def encode(self, text: str) -> np.ndarray:
		data, pos, ids = text.encode(), 0, []
		while pos < len(data):
			for b, i in self._by_len:
				if data.startswith(b, pos):
					ids.append(i)
					pos += len(b)
					break
		return np.array([ids], dtype=np.int64)

	def get_logits_from_input_ids(self, ids: list[int]) -> list[float]:
		self.calls.append(list(ids))
		# Safety valve: a broken loop must fail the test, not hang it.
		if len(self.calls) > 3000:
			raise RuntimeError("runaway decoding loop")
		return self.scorer(detok(ids))


def teacher(desired: bytes) -> Callable[[bytes], list[float]]:
	def score(ctx: bytes) -> list[float]:
		logits = np.full(SIZE, -5.0)
		rest = desired[len(ctx):] if desired.startswith(ctx) else b""
		for tid, tb in enumerate(TOKENS):
			if rest.startswith(tb):
				logits[tid] = float(len(tb))
		logits[PAD] = 100.0
		return logits.tolist()
	return score


def random_scorer(seed: int) -> Callable[[bytes], list[float]]:
	rng = np.random.default_rng(seed)

	def score(ctx: bytes) -> list[float]:
		logits = rng.standard_normal(SIZE)
		logits[PAD] = 50.0
		return logits.tolist()
	return score


def stubborn(favorite: bytes) -> Callable[[bytes], list[float]]:
	fav = TOKENS.index(favorite)
	rng = np.random.default_rng(0)

	def score(ctx: bytes) -> list[float]:
		logits = rng.standard_normal(SIZE) * 0.1
		logits[fav] = 20.0
		logits[PAD] = 50.0
		return logits.tolist()
	return score


# --------------------------------------------------------------------------
# Functions under test: every parameter type, every "last parameter" kind,
# a no-parameter function, and a name that is a prefix of another.
# --------------------------------------------------------------------------

def fn(fname: str, desc: str, /, **params: str) -> FunctionDef:
	return FunctionDef.model_validate({
		"name": fname, "description": desc,
		"parameters": {k: {"type": t} for k, t in params.items()},
	})


FUNCS = [
	fn("fn_add", "Add one to an integer.", x="integer"),
	fn("fn_add_numbers", "Add two numbers together.", a="number", b="number"),
	fn("fn_greet", "Greet a person by name.", name="string"),
	fn("fn_set_flag", "Set a flag.", name="string", enabled="boolean",
		level="integer", ratio="number"),
	fn("fn_toggle", "Turn something on or off.", on="boolean"),
	fn("fn_now", "Return the time."),
]
PREFIX = build_prefix(FUNCS).encode()


def make(functions: list[FunctionDef], scorer: Callable[..., Any],
	**kw: Any) -> tuple[Decoder, FakeModel]:
	model = FakeModel(scorer)
	return Decoder(model=model, vocab=VOCAB, functions=functions, **kw), model


# --------------------------------------------------------------------------
# 1. Teacher: the output is exactly the canonical call
# --------------------------------------------------------------------------

CASES: list[tuple[str, str, dict[str, Any]]] = [
	("What is the sum of 2 and 3?", "fn_add_numbers", {"a": 2.0, "b": 3.0}),
	("sum -1.5 and 40", "fn_add_numbers", {"a": -1.5, "b": 40.0}),
	("add 5", "fn_add", {"x": 5}),
	("add -12", "fn_add", {"x": -12}),
	("Greet shrek", "fn_greet", {"name": "shrek"}),
	("Greet café 😀", "fn_greet", {"name": "café 😀"}),
	('Greet "bob"', "fn_greet", {"name": "bob"}),
	("two\nlines\tand \\ slash", "fn_greet", {"name": "x"}),
	("", "fn_greet", {"name": ""}),
	("flag on", "fn_set_flag",
		{"name": "x", "enabled": True, "level": 3, "ratio": 0.5}),
	("flag off", "fn_set_flag",
		{"name": "", "enabled": False, "level": -7, "ratio": 10.0}),
	("toggle", "fn_toggle", {"on": True}),
	("toggle off", "fn_toggle", {"on": False}),
	("what time is it", "fn_now", {}),
]


@pytest.mark.parametrize("prompt, name, params", CASES)
def test_teacher_gets_exactly_the_canonical_call(
	prompt: str, name: str, params: dict[str, Any],
) -> None:
	want = call_text(prompt, name, params)
	dec, _ = make(FUNCS, teacher(PREFIX + want.encode()))
	res = dec.decode(prompt)
	assert isinstance(res, OutputResult)
	# call_text tells 2 from 2.0 and true from 1, unlike ==
	assert call_text(res.prompt, res.name, res.parameters) == want
	assert detok(dec.last_ids) == PREFIX + want.encode()


def test_function_name_that_prefixes_another_can_be_chosen() -> None:
	for name, params in (("fn_add", {"x": 1}),
			("fn_add_numbers", {"a": 1.0, "b": 2.0})):
		want = call_text("p", name, params)
		dec, _ = make(FUNCS, teacher(PREFIX + want.encode()))
		assert dec.decode("p").name == name


# --------------------------------------------------------------------------
# 2. Random model: whatever it wants, the call is valid
# --------------------------------------------------------------------------

def check_types(res: OutputResult, f: FunctionDef) -> None:
	assert res.name == f.name
	assert list(res.parameters) == list(f.parameters)
	for key, spec in f.parameters.items():
		v = res.parameters[key]
		if spec.type == "string":
			assert type(v) is str
			assert '"' not in v and "\\" not in v
			assert all(ord(ch) >= 0x20 for ch in v)
		elif spec.type == "integer":
			assert type(v) is int
		elif spec.type == "number":
			assert type(v) is float and math.isfinite(v)
		else:
			assert type(v) is bool


@pytest.mark.parametrize("f", FUNCS, ids=lambda f: f.name)
@pytest.mark.parametrize("seed", range(10))
def test_random_model_always_produces_a_valid_call(
	f: FunctionDef, seed: int,
) -> None:
	dec, _ = make([f], random_scorer(seed))
	res = dec.decode('request "q" é')
	check_types(res, f)
	assert res.prompt == 'request "q" é'
	assert all(i < len(TOKENS) for i in dec.last_ids), "padding id chosen"
	pre = build_prefix([f]).encode() + head('request "q" é').encode()
	assert detok(dec.last_ids).startswith(pre)


def test_random_model_over_the_whole_set_picks_only_listed_names() -> None:
	names = {f.name for f in FUNCS}
	seen = set()
	for seed in range(40):
		dec, _ = make(FUNCS, random_scorer(seed))
		seen.add(dec.decode("p").name)
	assert seen <= names
	assert len(seen) >= 3


# --------------------------------------------------------------------------
# 3. Stubborn model: only the cap can end a value
# --------------------------------------------------------------------------

def test_string_cap() -> None:
	dec, _ = make([FUNCS[2]], stubborn(b"x"), max_value_tokens=5)
	assert dec.decode("p").parameters["name"] == "x" * 5


def test_number_cap() -> None:
	dec, _ = make([FUNCS[1]], stubborn(b"7"), max_value_tokens=5)
	assert dec.decode("p").parameters == {"a": 77777.0, "b": 77777.0}


def test_integer_cap() -> None:
	dec, _ = make([FUNCS[0]], stubborn(b"7"), max_value_tokens=4)
	res = dec.decode("p")
	assert res.parameters == {"x": 7777} and type(res.parameters["x"]) is int


def test_cap_never_strands_a_half_written_number() -> None:
	# The model starts every number with "-". At the cap the number can't
	# end yet ("-"), so the decoder must finish it with a digit.
	dec, _ = make([FUNCS[1]], stubborn(b"-"), max_value_tokens=1)
	res = dec.decode("p")
	for v in res.parameters.values():
		assert type(v) is float and math.isfinite(v)


def test_stubborn_dot_cannot_make_a_malformed_number() -> None:
	dec, _ = make([FUNCS[1]], stubborn(b"."), max_value_tokens=3)
	check_types(dec.decode("p"), FUNCS[1])


def test_default_cap_stops_a_runaway_model() -> None:
	dec, model = make([FUNCS[2]], stubborn(b"x"))
	res = dec.decode("p")
	assert 0 < len(res.parameters["name"]) < 1000
	assert len(model.calls) < 1000


# --------------------------------------------------------------------------
# 4. Cost and hygiene
# --------------------------------------------------------------------------

def test_literal_scaffold_costs_no_forward_pass() -> None:
	f = FUNCS[5]                      # fn_now: only the name is a decision
	want = call_text("what time", "fn_now", {})
	dec, model = make([f], teacher(build_prefix([f]).encode() + want.encode()))
	before = len(model.calls)         # a probe in __init__ is fine
	dec.decode("what time")
	used = len(model.calls) - before
	name_steps = len(model.encode('fn_now"')[0])
	assert used <= name_steps + 1


def test_decode_leaves_no_state_behind() -> None:
	want = {
		"A": call_text("A", "fn_add", {"x": 1}),
		"B": call_text("B", "fn_greet", {"name": "bo"}),
	}

	def by_prompt(ctx: bytes) -> list[float]:
		tail = ctx[len(PREFIX):]
		key = "A" if tail.startswith(head("A").encode()) else "B"
		return teacher(PREFIX + want[key].encode())(ctx)

	dec, _ = make(FUNCS, by_prompt)
	first = dec.decode("A")
	assert dec.decode("B").name == "fn_greet"
	assert dec.decode("A") == first
	assert detok(dec.last_ids) == PREFIX + want["A"].encode()


# --------------------------------------------------------------------------
# 5. Function sets that must be refused up front
# --------------------------------------------------------------------------

def test_empty_function_list_is_refused() -> None:
	with pytest.raises(DecoderError):
		make([], random_scorer(0))


def test_duplicate_function_names_are_refused() -> None:
	with pytest.raises(DecoderError):
		make([FUNCS[2], FUNCS[2]], random_scorer(0))


@pytest.mark.parametrize("bad", ["object", "array"])
def test_unsupported_parameter_types_are_refused(bad: str) -> None:
	with pytest.raises(DecoderError):
		make([fn("fn_bad", "d", payload=bad)], random_scorer(0))