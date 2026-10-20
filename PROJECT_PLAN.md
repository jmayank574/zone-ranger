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

### 2026-10-20 — Phase 0 follow-ups (before Phase 1)

7. **Wait calibration is Uber-only in all months** (Lyft `on_scene_datetime` only exists from
   2025-03-06). Data card must state that waits are observed for completed trips only (cancelled
   requests are not in the data), so the simulator's patience/abandonment threshold is an
   assumption informed by the wait tail, not a measured value. Phase 2: quick check comparing Lyft
   vs Uber wait distributions for Mar 6 – Apr.
8. **Fares <= 0 or >= 500 are not dropped.** Keep the row, null the fare, set `fare_invalid` (same
   pattern as `wait_invalid`). Exclude flagged rows from fare-based work only (value model, revenue
   metrics). Phase 1: break down zero-fare rows by day, license and `driver_pay` and report what
   explains the March spike (64k rows with fare <= 0 vs 1.7k in Feb).
9. **Demand model forecasts full (unsampled) servable Manhattan demand at request time** (pickup
   and dropoff both in the geofence). The 5% sampling applies only inside the simulator.
