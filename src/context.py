# build prompt text (menu + question)
from __future__ import annotations

import json
from typing import Any, Iterable

from src.models import FunctionDef


# Reference text
def call_text(prompt: str, name: str, params: dict[str, Any]) -> str:
    """Canonical text of one function call, matching the output format"""
    return json.dumps(
        {"prompt": prompt, "name": name, "parameters": params},
        ensure_ascii=False,
    )


def _escape_json_string(s: str) -> str:
    """json.dumps of a string returns '"..."', slice off the outer quotes"""
    return json.dumps(s, ensure_ascii=False)[1:-1]


def head(prompt: str) -> str:
    """The exact prefix of ``call_text`` up to (and excluding) the name"""
    return '{"prompt": "' + _escape_json_string(prompt) + '", "name": "'


# Static pieces
INSTRUCTIONS = (
    "You are a function-calling engine.\n"
    "Given a request, pick exactly one function and extract its arguments.\n"
    "Reply with a single JSON object and nothing else.\n"
    "Available functions:"
)

FEW_SHOT: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("What is the sum of 2 and 3?", "fn_add_numbers", {"a": 2.0, "b": 3.0}),
    ("Greet shrek", "fn_greet", {"name": "shrek"}),
    ("Reverse the string 'hello'", "fn_reverse_string", {"s": "hello"}),
)


def format_function(fn: FunctionDef) -> str:
    """One line per function: ``- name(a: number, b: number): description``."""
    params = ", ".join(f"{k}: {v.type}" for k, v in fn.parameters.items())
    desc = " ".join(fn.description.split())
    return f"- {fn.name}({params}): {desc}"


def render_examples(
    examples: Iterable[tuple[str, str, dict[str, Any]]],
) -> str:
    """Render few-shot examples as ``Request: ...`` + ``call_text`` pairs."""
    return "\n".join(
        call_text(prompt, name, params)
        for prompt, name, params in examples
    )


def build_prefix(functions: list[FunctionDef]) -> str:
    """The prompt-independent part of the prompt. Cache once per run."""
    menu = "\n".join(format_function(fn) for fn in functions)
    examples = render_examples(FEW_SHOT)
    return (
        f"{INSTRUCTIONS}\n"
        f"{menu}\n"
        f"\n"
        f"Examples:\n"
        f"{examples}"
        f"\n"
    )
