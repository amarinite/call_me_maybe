# entry point: argparse, orchestration
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_sdk import Small_LLM_Model  # type: ignore

from src.decoder import Decoder, DecoderError
from src.loader import (
    LoadingError,
    load_func_definitions,
    load_user_prompts,
)
from src.models import OutputResult
from src.vocab import Vocab


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="call_me_maybe",
        description="function calling system using constrained decoding",
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        help="Path to the functions definition JSON file"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="Path to the function calling tests"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="Path to the function calling results"
    )

    return parser


def write_results(path: Path, results: list[OutputResult]) -> None:
    """Serialize to text first, then write"""
    payload = [r.model_dump() for r in results]
    text = json.dumps(payload, indent=4, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        functions = load_func_definitions(args.functions_definition)
        prompts = load_user_prompts(args.input)
    except LoadingError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    try:
        model = Small_LLM_Model()
        vocab = Vocab.from_file(model.get_path_to_vocab_file())
        decoder = Decoder(model=model, vocab=vocab, functions=functions)
    except DecoderError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as e:
        print(f"error: failed to initialise decoder: {e}", file=sys.stderr)
        return 1

    results: list[OutputResult] = []
    failures: list[tuple[int, str, str]] = []
    total = len(prompts)

    for index, item in enumerate(prompts, start=1):
        print(
            f"[{index}/{total}] {item.prompt!r}",
            file=sys.stderr,
            flush=True,
        )
        try:
            result = decoder.decode(item.prompt)
            results.append(result)
            print(
                f"    -> {result.name}  {list(result.parameters)}",
                file=sys.stderr,
                flush=True,
            )
        except DecoderError as e:
            failures.append((index, item.prompt, str(e)))
            print(f"    -> skipped: {e}", file=sys.stderr, flush=True)
        except Exception as e:
            failures.append((index, item.prompt, repr(e)))
            print(
                f"    -> skipped (unexpected): {e!r}",
                file=sys.stderr,
                flush=True,
            )

    try:
        write_results(args.output, results)
    except OSError as e:
        print(f"error: cannot write {args.output}: {e}", file=sys.stderr)
        return 1

    print(f"wrote {len(results)}/{total} result(s) to {args.output}")

    if failures:
        print(f"{len(failures)} prompt(s) failed:", file=sys.stderr)
        for index, prompt, reason in failures:
            print(f"  #{index}: {prompt!r} -> {reason}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
