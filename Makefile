.PHONY: install download dev-download eda features train replay experiment test lint

install:
	uv sync --extra dev

download:
	uv run fleetopt-download

dev-download:
	uv run fleetopt-download --dev

eda:
	uv run fleetopt-eda

features:
	uv run fleetopt-features

train:
	uv run fleetopt-train

replay:
	uv run fleetopt-replay

experiment:
	uv run fleetopt-experiment

test:
	uv run pytest --cov=fleetopt

lint:
	uv run ruff check .
	uv run ruff format --check .
