import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from src.loader import (
    LoadingError,
    load_func_definitions,
    load_user_prompts,
)

Loader = Callable[[Path], list[Any]]

VALID_FUNC: dict[str, Any] = {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
    "returns": {"type": "number"},
}


def write_json(path: Path, data: Any) -> Path:
    """Write ``data`` as JSON to ``path`` and return the path."""
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture(
    params=[load_func_definitions, load_user_prompts],
    ids=["func_definitions", "user_prompts"],
)
def loader(request: pytest.FixtureRequest) -> Loader:
    """Both public loaders; tests using this run once per loader."""
    loader_fn: Loader = request.param
    return loader_fn


# ------------------------------------------- shared behaviour (_load_list)


def test_missing_file(loader: Loader, tmp_path: Path) -> None:
    with pytest.raises(LoadingError, match="cannot read"):
        loader(tmp_path / "does_not_exist.json")


def test_path_is_a_directory(loader: Loader, tmp_path: Path) -> None:
    with pytest.raises(LoadingError, match="cannot read"):
        loader(tmp_path)


@pytest.mark.parametrize("text", ["", "[{", '[{"a": 1},]', "not json"])
def test_invalid_json(loader: Loader, tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(LoadingError, match="invalid JSON"):
        loader(path)


def test_file_is_not_utf8(loader: Loader, tmp_path: Path) -> None:
    path = tmp_path / "binary.json"
    path.write_bytes(b"\xff\xfe\x00garbage")
    with pytest.raises(LoadingError, match="UTF-8"):
        loader(path)


@pytest.mark.parametrize("payload", [{"k": "v"}, 5, "text", None])
def test_top_level_not_a_list(
    loader: Loader, tmp_path: Path, payload: Any
) -> None:
    path = write_json(tmp_path / "x.json", payload)
    with pytest.raises(LoadingError, match="expected a JSON array"):
        loader(path)


@pytest.mark.parametrize("item", ["a string", 3, None, []])
def test_item_not_an_object(
    loader: Loader, tmp_path: Path, item: Any
) -> None:
    path = write_json(tmp_path / "x.json", [item])
    with pytest.raises(LoadingError, match="invalid"):
        loader(path)


# ------------------------------------------------ load_user_prompts only


def test_prompts_valid(tmp_path: Path) -> None:
    path = write_json(tmp_path / "p.json", [{"prompt": "a"}, {"prompt": "b"}])
    assert [p.prompt for p in load_user_prompts(path)] == ["a", "b"]


def test_prompts_edge_case_strings(tmp_path: Path) -> None:
    texts = ["", 'say "hi" \\ \n ñ 日本語 🙂']
    path = write_json(tmp_path / "p.json", [{"prompt": t} for t in texts])
    assert [p.prompt for p in load_user_prompts(path)] == texts


@pytest.mark.parametrize("item", [{"text": "oops"}, {"prompt": 123}])
def test_prompts_invalid_item(tmp_path: Path, item: dict[str, Any]) -> None:
    path = write_json(tmp_path / "p.json", [item])
    with pytest.raises(LoadingError, match="invalid prompts"):
        load_user_prompts(path)


# -------------------------------------------- load_func_definitions only


def test_funcs_valid(tmp_path: Path) -> None:
    path = write_json(tmp_path / "f.json", [VALID_FUNC])
    funcs = load_func_definitions(path)
    assert funcs[0].name == "fn_add_numbers"
    assert funcs[0].parameters["a"].type == "number"


def test_funcs_parameter_order_preserved(tmp_path: Path) -> None:
    params = {n: {"type": "string"} for n in ["z", "a", "m"]}
    func = {"name": "f", "description": "d", "parameters": params}
    path = write_json(tmp_path / "f.json", [func])
    assert list(load_func_definitions(path)[0].parameters) == ["z", "a", "m"]


def test_funcs_without_parameters(tmp_path: Path) -> None:
    func = {"name": "fn_now", "description": "d", "parameters": {}}
    path = write_json(tmp_path / "f.json", [func])
    assert load_func_definitions(path)[0].parameters == {}


@pytest.mark.parametrize("missing", ["name", "description", "parameters"])
def test_funcs_missing_required_field(tmp_path: Path, missing: str) -> None:
    func = {k: v for k, v in VALID_FUNC.items() if k != missing}
    path = write_json(tmp_path / "f.json", [func])
    with pytest.raises(LoadingError, match="invalid function"):
        load_func_definitions(path)


@pytest.mark.parametrize(
    "parameters", [{"a": {"type": "banana"}}, ["a", "b"], {"a": "number"}]
)
def test_funcs_bad_parameters(tmp_path: Path, parameters: Any) -> None:
    func = {**VALID_FUNC, "parameters": parameters}
    path = write_json(tmp_path / "f.json", [func])
    with pytest.raises(LoadingError, match="invalid function"):
        load_func_definitions(path)


def test_funcs_error_reports_item_index(tmp_path: Path) -> None:
    path = write_json(tmp_path / "f.json", [VALID_FUNC, {"name": "x"}])
    with pytest.raises(LoadingError, match=r"1\.name|1\.description|1"):
        load_func_definitions(path)
