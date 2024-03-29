autoformat:
	set -e
	isort .
	black --config pyproject.toml .
	flake8

lint:
	set -e
	isort -c .
	black --check --config pyproject.toml .
	flake8

test:
	set -e
	coverage run -m pytest tests/

dev:
	pip install black coverage isort flake8 flake8-bugbear flake8-comprehensions

all: autoformat test build-docs