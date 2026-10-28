# load + validate input JSON files
import json
from pathlib import Path
from typing import Annotated, Any
from pydantic import Field, TypeAdapter, ValidationError

from src.models import FunctionDef, PromptItem


class LoadingError(Exception):
    """Raised when the on-disk JSON file can't be read, parsed or validated"""


def _read_json(path: Path, label: str) -> Any:
    """Shared core for reading and parsing raw JSON from disk."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except OSError as e:
        raise LoadingError(f"cannot read {label} from {path}: {e}") from e
    except UnicodeDecodeError as e:
        raise LoadingError(f"{path} is not valid UTF-8: {e}") from e
    except json.JSONDecodeError as e:
        raise LoadingError(f"invalid JSON in {path}: {e}") from e


def _load_list[T](
    path: Path, adapter: TypeAdapter[list[T]], label: str
) -> list[T]:
    data = _read_json(path, label)
    if not isinstance(data, list):
        raise LoadingError(
            f"{path}: expected a JSON array, got {type(data).__name__}"
        )
    try:
        return adapter.validate_python(data)
    except ValidationError as e:
        raise LoadingError(f"invalid {label} in {path}:\n{e}") from e

_FUNC_DEFS_ADAPTER: TypeAdapter[list[FunctionDef]] = TypeAdapter(
    list[FunctionDef]
)
_PROMPTS_ADAPTER: TypeAdapter[list[PromptItem]] = TypeAdapter(
    list[PromptItem]
)

_VOCAB_ADAPTER: TypeAdapter[
    dict[str, Annotated[int, Field(strict=True, ge=0)]]
] = TypeAdapter(
    dict[str, Annotated[int, Field(strict=True, ge=0)]]
)


def load_func_definitions(path: Path) -> list[FunctionDef]:
    return _load_list(path, _FUNC_DEFS_ADAPTER, "function definitions")


def load_user_prompts(path: Path) -> list[PromptItem]:
    return _load_list(path, _PROMPTS_ADAPTER, "prompts")


def load_vocab_mapping(path: Path) -> dict[str, int]:
    label = "vocabulary"
    data = _read_json(path, label)
    if not isinstance(data, dict):
        raise LoadingError(
            f"{path}: expected a JSON object, got {type(data).__name__}"
        )
    try:
        return _VOCAB_ADAPTER.validate_python(data)
    except ValidationError as e:
        raise LoadingError(f"invalid {label} in {path}:\n{e}") from e
