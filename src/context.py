# build prompt text (menu + question)
from __future__ import annotations

import json
from typing import Any

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
SELECT_INSTRUCTIONS = (
    "You are a function-calling engine.\n"
    "Pick the one function that does what the request asks.\n"
    "Available functions:"
)

ARGS_INSTRUCTIONS = (
    "You are a function-calling engine.\n"
    "Extract the arguments from the request. "
    "Copy values exactly as written in the request.\n"
    "Function:"
)


def format_function(fn: FunctionDef) -> str:
    """One line per function: ``- name(a: number, b: number): description``."""
    params = ", ".join(f"{k}: {v.type}" for k, v in fn.parameters.items())
    desc = " ".join(fn.description.split())
    return f"- {fn.name}({params}): {desc}"


def build_select_prefix(functions: list[FunctionDef]) -> str:
    """Prompt for choosing the function: the full menu."""
    menu = "\n".join(format_function(fn) for fn in functions)
    return f"{SELECT_INSTRUCTIONS}\n{menu}\n\n"


def build_args_prefix(fn: FunctionDef) -> str:
    """Prompt for extracting arguments: only the chosen function."""
    return f"{ARGS_INSTRUCTIONS}\n{format_function(fn)}\n\n"
