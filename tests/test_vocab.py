# """Tests for src.vocab and the vocab loading path in src.loader.

# Contract: the vocab file comes from the HF cache and is trusted. Vocab does
# NOT validate token strings, duplicate ids or an empty vocab. The only
# validation is the shape of the JSON, done by load_vocab_mapping and reported
# as LoadingError.

#     bytes_to_unicode() -> dict[int, str]       256 entries, byte -> char
#     unicode_to_bytes() -> dict[str, int]       the inverse
#     load_vocab_mapping(path) -> dict[str, int] LoadingError on bad file/shape
#     Vocab.from_file(path: Path | str) -> Vocab
#     Vocab.token_bytes(id) -> bytes             KeyError if id is unknown
#     Vocab.token_text(id) -> str | None         None if not valid UTF-8 alone
#     Vocab.items() -> iterator of (id, bytes)
#     len(vocab), id in vocab, vocab.max_id
# """
# import json
# from pathlib import Path
# from typing import Any

# import pytest

# from src.loader import LoadingError, load_vocab_mapping
# from src.vocab import Vocab, bytes_to_unicode, unicode_to_bytes


# # --------------------------------------------------------------------------
# # byte <-> unicode alphabet
# # --------------------------------------------------------------------------

# def test_alphabet_is_a_bijection_over_all_bytes() -> None:
# 	fwd = bytes_to_unicode()
# 	assert sorted(fwd) == list(range(256))
# 	assert all(len(c) == 1 for c in fwd.values())
# 	assert len(set(fwd.values())) == 256
# 	back = unicode_to_bytes()
# 	assert all(back[c] == b for b, c in fwd.items())


# def test_alphabet_printable_ascii_is_identity() -> None:
# 	fwd = bytes_to_unicode()
# 	for b in range(ord("!"), ord("~") + 1):
# 		assert fwd[b] == chr(b)


# def test_alphabet_latin1_ranges_are_identity() -> None:
# 	fwd = bytes_to_unicode()
# 	for b in [*range(0xA1, 0xAD), *range(0xAE, 0x100)]:
# 		assert fwd[b] == chr(b)


# def test_alphabet_leftover_bytes_count_up_from_256() -> None:
# 	fwd = bytes_to_unicode()
# 	leftovers = [*range(0, 33), *range(127, 161), 173]
# 	assert len(leftovers) == 68
# 	for n, b in enumerate(leftovers):
# 		assert fwd[b] == chr(256 + n)


# def test_alphabet_known_special_cases() -> None:
# 	fwd = bytes_to_unicode()
# 	assert fwd[ord(" ")] == "\u0120"  # G with dot above
# 	assert fwd[ord("\n")] == "\u010a"  # C with dot above


# # --------------------------------------------------------------------------
# # Vocab on a tiny fake vocab
# # --------------------------------------------------------------------------

# FAKE: dict[str, int] = {
# 	"a": 0,
# 	"\u0120b": 1,            # " b"
# 	"\u010a": 2,             # "\n"
# 	"\u00c3\u00a9": 3,       # "é" as one token (bytes c3 a9)
# 	"\u00c3": 4,             # only the first byte of "é"
# 	"\u00a9": 5,             # only the second byte of "é"
# 	"\u00f0\u0141\u013a\u0122": 6,  # "😀" (4 bytes)
# 	"\"": 7,
# 	"\u0120": 9,             # id 8 is missing on purpose
# }


# def _write(tmp_path: Path, content: str | bytes) -> Path:
# 	p = tmp_path / "vocab.json"
# 	if isinstance(content, bytes):
# 		p.write_bytes(content)
# 	else:
# 		p.write_text(content, encoding="utf-8")
# 	return p


# @pytest.fixture
# def fake_path(tmp_path: Path) -> Path:
# 	return _write(tmp_path, json.dumps(FAKE, ensure_ascii=False))


# @pytest.fixture
# def fake_vocab(fake_path: Path) -> Vocab:
# 	return Vocab.from_file(fake_path)


# def test_from_file_accepts_str_and_path(fake_path: Path) -> None:
# 	assert len(Vocab.from_file(fake_path)) == len(FAKE)
# 	assert len(Vocab.from_file(str(fake_path))) == len(FAKE)


# def test_basic_lookup(fake_vocab: Vocab) -> None:
# 	assert fake_vocab.token_bytes(0) == b"a"
# 	assert fake_vocab.token_bytes(1) == b" b"
# 	assert fake_vocab.token_text(1) == " b"
# 	assert fake_vocab.token_bytes(2) == b"\n"
# 	assert fake_vocab.token_bytes(7) == b'"'
# 	assert fake_vocab.token_bytes(9) == b" "


# def test_multibyte_token(fake_vocab: Vocab) -> None:
# 	assert fake_vocab.token_bytes(3) == "é".encode()
# 	assert fake_vocab.token_text(3) == "é"
# 	assert fake_vocab.token_bytes(6) == "😀".encode()
# 	assert fake_vocab.token_text(6) == "😀"


# def test_partial_utf8_tokens_keep_bytes_but_have_no_text(
# 	fake_vocab: Vocab,
# ) -> None:
# 	assert fake_vocab.token_bytes(4) == b"\xc3"
# 	assert fake_vocab.token_bytes(5) == b"\xa9"
# 	assert fake_vocab.token_text(4) is None
# 	assert fake_vocab.token_text(5) is None
# 	joined = fake_vocab.token_bytes(4) + fake_vocab.token_bytes(5)
# 	assert joined.decode("utf-8") == "é"


# def test_size_membership_and_gaps(fake_vocab: Vocab) -> None:
# 	assert len(fake_vocab) == len(FAKE)
# 	assert fake_vocab.max_id == 9
# 	assert 8 not in fake_vocab
# 	assert 9 in fake_vocab


# def test_unknown_id_raises_keyerror(fake_vocab: Vocab) -> None:
# 	with pytest.raises(KeyError):
# 		fake_vocab.token_bytes(8)
# 	with pytest.raises(KeyError):
# 		fake_vocab.token_text(8)


# def test_items_covers_every_token(fake_vocab: Vocab) -> None:
# 	got = dict(fake_vocab.items())
# 	assert set(got) == set(FAKE.values())
# 	assert got[3] == "é".encode()


# def test_vocab_is_immutable(fake_vocab: Vocab) -> None:
# 	with pytest.raises(Exception):
# 		fake_vocab.id_to_bytes = {}  # type: ignore[misc]


# # --------------------------------------------------------------------------
# # Loading: the only validation left is the JSON shape (LoadingError)
# # --------------------------------------------------------------------------

# def test_load_vocab_mapping_returns_raw_mapping(fake_path: Path) -> None:
# 	assert load_vocab_mapping(fake_path) == FAKE


# def test_missing_file(tmp_path: Path) -> None:
# 	with pytest.raises(LoadingError):
# 		load_vocab_mapping(tmp_path / "nope.json")


# def test_path_is_a_directory(tmp_path: Path) -> None:
# 	with pytest.raises(LoadingError):
# 		load_vocab_mapping(tmp_path)


# def test_not_utf8(tmp_path: Path) -> None:
# 	with pytest.raises(LoadingError):
# 		load_vocab_mapping(_write(tmp_path, b"\xff\xfe{"))


# @pytest.mark.parametrize(
# 	"content",
# 	[
# 		"{not json",
# 		"",
# 		"[1, 2, 3]",          # wrong top-level type
# 		'{"a": "zero"}',      # id is not an int
# 		'{"a": true}',        # strict mode: bool is not an id
# 		'{"a": 1.5}',         # float id
# 		'{"a": -1}',          # negative id
# 	],
# )
# def test_invalid_vocab_shape(tmp_path: Path, content: str) -> None:
# 	with pytest.raises(LoadingError):
# 		load_vocab_mapping(_write(tmp_path, content))


# def test_from_file_propagates_loading_error(tmp_path: Path) -> None:
# 	with pytest.raises(LoadingError):
# 		Vocab.from_file(tmp_path / "nope.json")


# # --------------------------------------------------------------------------
# # Real model (skipped when the model can't be loaded, e.g. offline CI)
# # Run with `pytest -rs` to see whether these were skipped.
# # --------------------------------------------------------------------------

# @pytest.fixture(scope="module")
# def real() -> Any:
# 	try:
# 		from llm_sdk import Small_LLM_Model
# 		model = Small_LLM_Model()
# 		vocab = Vocab.from_file(model.get_path_to_vocab_file())
# 	except Exception as e:  # noqa: BLE001
# 		pytest.skip(f"real model unavailable: {e}")
# 	return model, vocab


# ROUNDTRIP_TEXTS = [
# 	"hello",
# 	" hello",
# 	"What is the sum of 40 and 2?",
# 	"Greet shrek",
# 	"Reverse the string 'hello'",
# 	'say "hi"',
# 	"line1\nline2",
# 	"tab\there",
# 	"  double  space",
# 	"café",
# 	"日本語",
# 	"😀",
# 	"3.14159 -42 1e10",
# ]


# @pytest.mark.parametrize("text", ROUNDTRIP_TEXTS)
# def test_real_tokenizer_roundtrip(real: Any, text: str) -> None:
# 	model, vocab = real
# 	ids = model.encode(text)[0].tolist()
# 	assert b"".join(vocab.token_bytes(i) for i in ids) == text.encode()


# def test_real_vocab_loads_with_partial_utf8_tokens(real: Any) -> None:
# 	_, vocab = real
# 	partial = [i for i, _ in vocab.items() if vocab.token_text(i) is None]
# 	assert partial, "expected some tokens that are not valid UTF-8 alone"


# def test_real_no_empty_tokens(real: Any) -> None:
# 	_, vocab = real
# 	assert all(len(b) > 0 for _, b in vocab.items())


# def test_real_structural_single_char_tokens_exist(real: Any) -> None:
# 	_, vocab = real
# 	all_bytes = {b for _, b in vocab.items()}
# 	for ch in '0123456789-.,"{}:[] ':
# 		assert ch.encode() in all_bytes, f"no single token for {ch!r}"


# def test_real_logits_are_at_least_as_wide_as_vocab(real: Any) -> None:
# 	model, vocab = real
# 	ids = model.encode("hello")[0].tolist()
# 	logits = model.get_logits_from_input_ids(ids)
# 	assert len(logits) >= vocab.max_id + 1
