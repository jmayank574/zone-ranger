.PHONY: install download dev-download eda features train replay experiment test lint

install:
	uv sync --extra dev

download:
	uv run zone-ranger-download

dev-download:
	uv run zone-ranger-download --dev

eda:
	uv run zone-ranger-eda

features:
	uv run zone-ranger-features

train:
	uv run zone-ranger-train

replay:
	uv run zone-ranger-replay

experiment:
	uv run zone-ranger-experiment

test:
	uv run pytest --cov=zone_ranger

lint:
	uv run ruff check .
	uv run ruff format --check .
