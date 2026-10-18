.PHONY: install run debug clean lint lint-strict test

run:
	uv run python -m src

install:
	uv sync

debug:
	uv run python -m pdb -m src

clean:
	rm -rf .mypy_cache .pytest_cache __pycache__ src/__pycache__

lint:
	uv run flake8 src/ --exclude=.venv
	uv run mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src/ --exclude=.env
	uv run mypy src/ --strict

test:
	uv run pytest
