# Zone Ranger

Autonomous fleet dispatch and idle-vehicle rebalancing on NYC ride-hailing data.

Time-stepped fleet simulator over real NYC HVFHV requests, demand and ETA models, CP-SAT dispatch,
idle-fleet rebalancing, and offline policy experiments.

Status: Phase 1 done (features, ETA and demand models, travel matrix; see `docs/` and
`reports/evaluation.json`). Phase 2 (simulator and dispatch) is next. Results tables will be
generated from `reports/*.json`.

## Quick start

```
uv sync --extra dev
uv run zone-ranger-download --dev   # small dev month
uv run zone-ranger-features --dev
uv run zone-ranger-train --dev      # add --force to ignore stage checkpoints
uv run pytest --cov=zone_ranger
```

See `PROJECT_PLAN.md` for the design and phase plan.
