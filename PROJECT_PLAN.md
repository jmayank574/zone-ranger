# fleetopt project plan

The full kickoff plan (phases 0-6) lives in the original brief; this file tracks decisions made
along the way.

## Decisions

### 2026-10-20 — Phase 0

1. **Months / data:** Jan–Feb 2025 train, Mar val, Apr test. All four months downloaded and profiled.
2. **Geofence:** Manhattan, 69 zones (excludes 264/265). **Demand sampling:** 5%. Serve only requests
   whose pickup AND dropoff are inside the geofence; the demand model forecasts that servable demand.
3. **Fleet size:** config parameter `DEFAULT_FLEET_SIZE = 225`; sweep {150, 225, 300} later. 10%
   sampling is a later stress test.
4. **Cleaning:** drop invalid zones, base fare outside (0, 500), trip miles outside [0.1, 100], trip
   time outside [1, 180] min (plus structural checks: request present, pickup in file month,
   dropoff after pickup — all ~0 rows). Rows with request-after-pickup, on-scene-before-request, or
   pickup delay > 3 h are **kept** for demand and ETA; their wait fields are nulled and
   `wait_invalid = true` so wait calibration excludes them.
5. **Coverage:** `stubs.py` omitted from coverage until the stages exist.
6. **Storage:** everything on D:. `data/raw` (including `.part` files), `data/duckdb_tmp` (DuckDB
   spill dir, set on every connection via `fleetopt.db.connect`), the DuckDB file, `artifacts/`,
   `reports/` and `.venv` all resolve under `D:\FleetOp`. `UV_CACHE_DIR` moves to `D:\uv-cache`.
