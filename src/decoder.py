# generation loop (constraint + model + masking)
# plus the final validation with OutputResult
from __future__ import annotations

from typing import Any

import numpy as np

from src.constraint import (
    ChoiceConstraint,
    NumberConstraint,
    SharedMasks,
    StringConstraint,
    mask_logits,
)
from src.context import build_prefix, head
from src.models import FunctionDef, OutputResult
from src.vocab import Vocab
from llm_sdk import Small_LLM_Model  # type: ignore

_SUPPORTED_TYPES = ("string", "number", "integer", "boolean")


class DecoderError(Exception):
    """Bad configuration or impossible constrained-decoding state."""


class Decoder:
    """Constrained decoder for one JSON function-call output"""

    def __init__(
        self,
        model: Small_LLM_Model,
        vocab: Vocab,
        functions: list[FunctionDef],
        *,
        max_value_tokens: int = 48,
    ) -> None:
        self._model = model
        self._vocab = vocab
        self._cap = max_value_tokens

        self._validate_functions(functions)
        self._by_name: dict[str, FunctionDef] = {f.name: f for f in functions}

        self._size = self._logits_width()

        self.masks = SharedMasks.build(vocab, self._size)

        self._prefix_text: str = build_prefix(functions)
        self._prefix_ids: list[int] = self._encode(self._prefix_text)

        self._name_choice = ChoiceConstraint(
            options=tuple((f.name + '"').encode("utf-8") for f in functions),
            vocab=vocab,
            masks=self.masks,
        )
        self._bool_choice = ChoiceConstraint(
            options=(b"true", b"false"),
            vocab=vocab,
            masks=self.masks,
        )
        self._string = StringConstraint(masks=self.masks)
        self._num: dict[tuple[str, bool], NumberConstraint] = {}
        for as_type in ("int", "float"):
            for is_last in (True, False):
                self._num[(as_type, is_last)] = NumberConstraint(
                    masks=self.masks,
                    terminator=(
                        self.masks.close_brace if is_last
                        else self.masks.comma
                    ),
                    as_type=as_type,
                )

        # Filled by decode(); tests replay it to check the assembled text.
        self.last_ids: list[int] = []

    def decode(self, prompt: str) -> OutputResult:
        head_text = head(prompt)
        ids: list[int] = list(self._prefix_ids) + self._encode(head_text)

        name = self._run_name_choice(ids)
        ids.extend(self._encode(', "parameters": {'))

        schema = self._by_name[name].parameters
        params: dict[str, Any] = {}

        if not schema:
            ids.extend(self._encode("}}"))
        else:
            items = list(schema.items())
            for index, (pname, pspec) in enumerate(items):
                is_last = index == len(items) - 1

                key_literal = f'"{pname}": '
                if pspec.type == "string":
                    key_literal += '"'
                ids.extend(self._encode(key_literal))

                params[pname] = self._run_value(
                    pspec.type, is_last=is_last, ids=ids
                )

                if not is_last:
                    sep = ", " if pspec.type in ("string", "boolean") else " "
                    ids.extend(self._encode(sep))

            last_type = items[-1][1].type
            closing = "}" if last_type in ("number", "integer") else "}}"
            ids.extend(self._encode(closing))

        self.last_ids = ids
        return OutputResult(prompt=prompt, name=name, parameters=params)

    def _run_name_choice(self, ids: list[int]) -> str:
        buf = self._run_constraint(
            self._name_choice, ids=ids, closer=self.masks.quote,
        )
        return self._name_choice.value(buf)

    def _run_value(
        self, type_name: str, *, is_last: bool, ids: list[int]
    ) -> Any:
        """Run one parameter value; return the typed value."""
        if type_name == "string":
            buf = self._run_constraint(
                self._string, ids=ids, closer=self.masks.quote,
            )
            return self._string.value(buf)

        if type_name in ("number", "integer"):
            as_type = "int" if type_name == "integer" else "float"
            constraint = self._num[(as_type, is_last)]
            buf = self._run_constraint(
                constraint, ids=ids, closer=constraint.terminator,
            )
            return constraint.value(buf)

        if type_name == "boolean":
            buf = self._run_constraint(
                self._bool_choice, ids=ids, closer=None,
            )
            return self._bool_choice.value(buf) == "true"

    def _run_constraint(
        self,
        constraint: ChoiceConstraint | NumberConstraint | StringConstraint,
        *,
        ids: list[int],
        closer: np.ndarray | None,
    ) -> bytes:
        """Run a constraint to completion; extend ``ids`` in place."""

        buf = b""
        steps = 0

        while not constraint.is_complete(buf):
            allowed = constraint.allowed(buf)

            if steps >= self._cap and closer is not None:
                forced = allowed & closer
                if forced.any():
                    allowed = forced

            if not allowed.any():
                raise DecoderError(
                    f"no legal token at {buf!r} after {steps} step(s)"
                )

            logits = np.asarray(self._model.get_logits_from_input_ids(ids))
            tid = int(mask_logits(logits, allowed).argmax())
            ids.append(tid)
            buf += self._vocab.token_bytes(tid)
            steps += 1

        return buf

    def _encode(self, text: str) -> list[int]:
        """Encode text to a flat list of token ids."""
        return list(self._model.encode(text).tolist()[0])

    def _logits_width(self) -> int:
        """Probe the model's logit vector length (may exceed len(vocab))."""
        probe = self._encode("x")
        return len(self._model.get_logits_from_input_ids(probe))

    def _validate_functions(self, functions: list[FunctionDef]) -> None:
        if not functions:
            raise DecoderError("empty function list")
        seen: set[str] = set()
        for fn in functions:
            if fn.name in seen:
                raise DecoderError(f"duplicate function name {fn.name!r}")
            seen.add(fn.name)
            for pname, pspec in fn.parameters.items():
                if pspec.type not in _SUPPORTED_TYPES:
                    raise DecoderError(
                        f"function {fn.name!r} param {pname!r}: "
                        f"type {pspec.type!r} is not supported "
                        f"(allowed: {', '.join(_SUPPORTED_TYPES)})"
                    )
