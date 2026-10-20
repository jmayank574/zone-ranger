# Zone Ranger

Autonomous fleet dispatch and idle-vehicle rebalancing on NYC ride-hailing (HVFHV) data. Python
package `zone_ranger` (`src/zone_ranger`). Built as a portfolio project for a Waymo Product Data
Scientist, Marketplace Logistics role: ML for demand/ETA/vehicle value, CP-SAT dispatch and
rebalancing, offline policy experiments, online experiment design.

**Read `PROJECT_PLAN.md` first** (goals, principles, phase plan, decisions log).

## Status

Phase 0 (setup, EDA) and Phase 1 (features, models, travel matrix) are done. Next: Phase 2
(simulator, greedy + CP-SAT dispatch, API, Docker). Do not start a phase until the user says so.

Phase 1 results (April 2025 test window, model trained on Jan 8 - Feb 28, test scored once):
- ETA (LightGBM L1): MAE 3.26 min vs 3.88 for the route-median baseline (6.37 global median).
  The `trip_miles` ablation (post-trip quantity, not deployable) reaches 2.57.
- Demand (LightGBM Poisson, zone x 15 min, 1-step ahead): WMAPE 0.160 vs 0.239 for same-slot-last-week.
- Full numbers: `reports/evaluation.json`; write-ups in `docs/MODEL_CARD.md`, `docs/DATA_CARD.md`.
- `artifacts/` is gitignored and was deleted in the rename; run `zone-ranger-train` to rebuild.

## Commands

```
uv sync --extra dev
uv run zone-ranger-download [--dev]
uv run zone-ranger-eda
uv run zone-ranger-features [--dev]
uv run zone-ranger-train [--dev] [--force]   # stages checkpoint; --force retrains all
uv run python -m pytest --cov=zone_ranger
uv run ruff check . && uv run ruff format --check .
```

Launch long runs with `python -u` (e.g. `uv run python -u -m zone_ranger.train`) so logs stream.
Everything lives on D: (data, DuckDB spill dir, artifacts, `UV_CACHE_DIR=D:\uv-cache`).

## Rules

- One commit per milestone; run tests and ruff before committing.
- Update this file at the end of every phase (status, results, commands).
- Every reported number comes from checked-in code and `reports/*.json`; never invent results.
- Point-in-time correctness: features are fit on train only or lag the prediction time.
- No license file and no third-party credit lines in the repo.
