# entry point: argparse, orchestration
import argparse
import sys
from pathlib import Path

from src.loader import (
	LoadingError,
	load_func_definitions,
	load_user_prompts,
)


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


def main() -> int:
	parser = build_parser()
	args = parser.parse_args()

	# --- temporary test block ---
	try:
		functions = load_func_definitions(args.functions_definition)
		prompts = load_user_prompts(args.input)
	except LoadingError as e:
		print(f"error: {e}", file=sys.stderr)
		return 1

	print(f"Loaded {len(functions)} functs from {args.functions_definition}")
	for i, fn in enumerate(functions):
		print(f"  [{i}] {fn.name}  params={list(fn.parameters)}")
	print(f"Loaded {len(prompts)} prompts from {args.input}")
	for i, p in enumerate(prompts):
		print(f"  [{i}] {p.prompt!r}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
