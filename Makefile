.PHONY: install run debug clean fclean lint lint-strict test

UV_CACHE_DIR    ?= $(HOME)/.cache/uv
UV_PROJECT_ENV  ?= $(CURDIR)/.venv
HF_HOME         ?= $(HOME)/.cache/huggingface

run:
	uv run python -m src

install:
	uv sync

debug:
	uv run python -m pdb -m src

clean:
	rm -rf .mypy_cache .pytest_cache __pycache__ src/__pycache__ .coverage

fclean: clean
	rm -rf "$(UV_CACHE_DIR)"
	rm -rf "$(HF_HOME)"
	rm -rf "$(UV_PROJECT_ENV)"

lint:
	uv run flake8 src/ --exclude=.venv
	uv run mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src/ --exclude=.env
	uv run mypy src/ --strict

test:
	uv run pytest
