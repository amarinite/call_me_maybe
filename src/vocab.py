# token↔string maps from vocab file, helpers
#
# TODO: fix VocabError mess
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Any

from pydantic import BaseModel, ConfigDict, PrivateAttr
from src.loader import load_vocab_mapping


def bytes_to_unicode() -> dict[int, str]:
	"""Returns mapping from byte integer to unicode char for byte-level BPE."""
	safe_bytes = (
		list(range(ord("!"), ord("~") + 1))
		+ list(range(ord("¡"), ord("¬") + 1))
		+ list(range(ord("®"), ord("ÿ") + 1))
	)
	unicode_chars = safe_bytes[:]
	next_unicode_offset = 0
	for byte_val in range(256):
		if byte_val not in safe_bytes:
			safe_bytes.append(byte_val)
			unicode_chars.append(256 + next_unicode_offset)
			next_unicode_offset += 1

	return {
		byte_val: chr(unicode_value)
		for byte_val, unicode_value in zip(safe_bytes, unicode_chars)
	}


def unicode_to_bytes() -> dict[str, int]:
	"""Returns inverse mapping from unicode character to byte integer."""
	return {
		unicode_char: byte_val
		for byte_val, unicode_char in bytes_to_unicode().items()
	}


class Vocab(BaseModel):
	"""Manages model vocab, mapping token IDs to raw bytes and decoded text."""
	model_config = ConfigDict(
		extra="forbid",
		frozen=True
	)

	id_to_bytes: dict[int, bytes]
	_max_id: int = PrivateAttr(default=-1)
	_id_to_text: dict[int, str | None] = PrivateAttr(default_factory=dict)

	def model_post_init(self, __context: object) -> None:
		super().model_post_init(__context)
		self._max_id = max(self.id_to_bytes.keys()) if self.id_to_bytes else -1

		text_map: dict[int, str | None] = {}
		for tid, bz in self.id_to_bytes.items():
			try:
				text_map[tid] = bz.decode("utf-8")
			except UnicodeDecodeError:
				text_map[tid] = None
		self._id_to_text = text_map

	@property
	def max_id(self) -> int:
		"""Return the highest token ID in the vocabulary."""
		return self._max_id

	def token_bytes(self, token_id: int) -> bytes:
		"""Return exact raw bytes for a token ID."""
		return self.id_to_bytes[token_id]

	def token_text(self, token_id: int) -> str | None:
		"""Return decoded UTF-8 string if valid, otherwise None."""
		return self._id_to_text[token_id]

	def items(self) -> Iterator[tuple[int, bytes]]:
		"""Iterate over all (token_id, raw_bytes) pairs."""
		return iter(self.id_to_bytes.items())

	def __len__(self) -> int:
		return len(self.id_to_bytes)

	def __contains__(self, token_id: int) -> bool:
		return token_id in self.id_to_bytes

	@classmethod
	def from_file(cls, path: Path | str) -> Vocab:
		"""Load vocabulary from a JSON file."""
		vocab_dict = load_vocab_mapping(Path(path))
		u2b = unicode_to_bytes()
		id_to_bytes: dict[int, bytes] = {
			token_id: bytes([u2b[c] for c in token_str])
			for token_str, token_id in vocab_dict.items()
		}
		return cls(id_to_bytes=id_to_bytes)
