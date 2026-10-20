# Project Plan: Zone Ranger — autonomous fleet dispatch and idle-vehicle rebalancing

> Read this file first. It holds the goals, design notes, phase plan and the running decisions log.

---

## 1. Context for Claude Code

I'm Mayank, a USC MS Analytics grad targeting a **Product Data Scientist, Marketplace Logistics** role at Waymo. The role is about:

- optimization models that dispatch **off-hail (idle) vehicles to high-value locations**
- ML models for **vehicle value prediction** and **fleet utilization**
- **offline and online experiments** to evaluate model deployments
- turning algorithms into **production-grade pipelines**
- solver experience: **CP-SAT** (OR-Tools), CPLEX, Gurobi

I want to build an end-to-end marketplace ML + optimization system, from scratch, on **real NYC ride-hailing data**. I am building this to learn the full product-data-science workflow, so explain what you build and why as we go.

### Design notes

A system of this kind has these building blocks; each is a design choice we make ourselves.

**Pipeline design**
- **Features:** one DuckDB `CREATE TABLE trips AS SELECT ... FROM read_parquet(glob)` with cleaning filters and derived calendar columns. A `demand` table on a **complete zone x 15-min grid** (`range()` x `generate_series()` LEFT JOIN counts, so zero-demand intervals exist), with lags and rolling means from `LAG()` / `AVG() OVER (ROWS BETWEEN n PRECEDING AND 1 PRECEDING)` window functions.
- **Artifacts:** an `EtaFeatureEncoder` fits route/pickup/dropoff duration medians on train only (global-median fallback). `EtaBundle` / `DemandBundle` dataclasses hold encoder + model + version + metrics, saved with joblib and type-checked on load.
- **Training:** splits read from DuckDB (ETA train = seeded 2M-row reservoir sample; val/test full). LightGBM L1 for ETA, LightGBM Poisson for demand. `reports/evaluation.json` holds row counts, split dates, val + test metrics and a `limitations` list.
- **API and delivery:** a `create_app()` factory with artifacts loaded once in a `lifespan` handler from an env-var directory, a dedicated Prometheus registry (request Counter + latency Histogram per endpoint), 503 when artifacts are missing, model version in every response. Makefile targets per stage; CI runs `ruff check`, `ruff format --check`, `pytest --cov`, and `docker build`; Dockerfile is python-slim + `libgomp1` (LightGBM needs it) + non-root user; `reports/latency.json` from an in-process benchmark (20 warm-up + 500 measured requests).

### Design principles beyond the basics
1. **A real simulation.** A time-stepped simulator with persistent vehicle state, so repositioning, abandonment and utilization can be evaluated (independent static batches cannot).
2. **The demand model drives a decision.** Its forecast feeds rebalancing.
3. **Naive baselines for every ML model.** Seasonal-naive / median baselines are reported next to every model metric.
4. **CP-SAT** for dispatch and rebalancing: handles unequal vehicle/request counts without dummy padding and supports side constraints.
5. **Demand is timed at request.** HVFHV gives `request_datetime`, the true demand signal; demand is bucketed by request time.
6. **Zones 264/265** are "Unknown/Outside NYC" in the zone lookup and are excluded.
7. **Uncertainty, not single numbers.** Confidence intervals across days and seeds.

---

## 2. What we're building

**Phase A (baseline system):** ingest real ride-hailing data → point-in-time features → ETA + demand models → simulated robotaxi fleet replaying real requests → request-to-vehicle dispatch (greedy vs. CP-SAT) → API + metrics + tests + docs.

**Phase B (our enhancements, the Waymo-specific part):** vehicle-value model → CP-SAT **idle-fleet rebalancing** → offline policy experiment (control vs. treatment) with proper statistics → online experiment design.

```
TLC HVFHV Parquet ──> DuckDB ──┬──> ETA features ──────> ETA model ────────────┐
                               ├──> Demand features ───> Demand model ─────────┤
                               └──> Value features ────> Vehicle-value model ──┤
                                                                               v
     Real ride requests (test period) ──> Fleet simulator <── Rebalancer (CP-SAT, idle vehicles)
                                               │          <── Dispatcher (CP-SAT, requests)
                                               v
                                  Policy experiment: control vs treatment
                                               v
                               reports/*.json  ·  FastAPI  ·  Prometheus
```

---

## 3. Data

**Source:** NYC TLC **High Volume For-Hire Vehicle (HVFHV)** trip records (Uber/Lyft): `https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_YYYY-MM.parquet`, plus the taxi zone lookup CSV and taxi zone shapefile (for zone centroids and borough/geofence).

Why HVFHV over yellow taxi: it is ride-hailing (Waymo's business), and it has `request_datetime`, `on_scene_datetime`, `pickup_datetime`, which give **real passenger wait times** for calibrating the simulator, plus fares (`base_passenger_fare`), `trip_miles`, `trip_time`, and zone IDs.

**Inspect the actual schema first** (`DESCRIBE` in DuckDB). Don't assume column names.

**Scale (it's roughly 15-20M trips/month, 400-500 MB/file):**
- Default months: train **Jan-Feb 2025**, validation **Mar 2025**, test **Apr 2025**. Make months configurable in `constants.py`.
- Never load full months into pandas. Do all heavy work in DuckDB SQL; pull only aggregated or sampled data into Python.
- Add a `--dev` flag that runs every stage on a small sample (e.g. 1 week, 1% of trips) so the full pipeline can be smoke-tested in minutes.
- **Service area:** model a Waymo-style geofence (default: Manhattan, configurable list of zone IDs) and a **demand sampling rate** (e.g. 5-10% of requests in the geofence) so a simulated fleet of a few hundred vehicles is realistic and tractable.

**Cleaning:** drop missing/invalid zones, non-positive or extreme durations/distances/fares, timestamps outside the file's month, request after pickup, etc. Log row counts at each step into `reports/data_quality.json`.

---

## 4. Non-negotiable principles

1. **Point-in-time correctness.** Any statistic used as a feature (zone medians, route travel times, demand lags) is fit on the training window only, or lags the prediction time. Val/test strictly after train. Test month used only for final evaluation and simulation.
2. **Honest leakage notes.** `trip_miles` and `trip_time` are only known after a trip. If the ETA model uses `trip_miles` as a stand-in for a routing-engine distance estimate, document that explicitly in the model card.
3. **Every reported number is generated by checked-in code** and written to `reports/*.json`. The README table is filled from those files, never typed by hand.
4. **Baselines for everything:** greedy dispatch vs. CP-SAT; no-rebalancing vs. rebalancing; naive seasonal demand forecast vs. LightGBM.
5. **Reproducibility:** fixed seeds, config in one place, one CLI command per stage.

---

## 5. Repo layout (target)

```
fleetopt/
├── src/fleetopt/
│   ├── constants.py       # months, splits, geofence, paths, seeds
│   ├── download.py        # fetch Parquet + zone lookup/shapefile
│   ├── features.py        # DuckDB: clean trips, ETA/demand/value feature tables
│   ├── train.py           # LightGBM ETA + demand (+ value in Phase B), temporal eval
│   ├── travel.py          # zone-to-zone travel time/distance matrix (train-window medians)
│   ├── simulator.py       # time-stepped fleet simulator over real requests
│   ├── dispatch.py        # greedy + CP-SAT request assignment
│   ├── rebalance.py       # (Phase B) CP-SAT idle-vehicle repositioning
│   ├── value.py           # (Phase B) vehicle-value targets + model
│   ├── experiment.py      # (Phase B) policy comparison, paired stats, sweeps
│   ├── replay.py          # CLI: run simulations, write reports
│   ├── api.py, schemas.py # FastAPI + Pydantic
│   └── metrics.py         # Prometheus + marketplace metric definitions
├── tests/                 # unit + small integration tests on synthetic data
├── docs/ARCHITECTURE.md, MODEL_CARD.md, DATA_CARD.md, EXPERIMENT_DESIGN.md
├── reports/               # machine-readable results (JSON), committed
├── data/, artifacts/      # gitignored
├── pyproject.toml, Makefile, Dockerfile, docker-compose.yml, .github/workflows/ci.yml
└── README.md
```

CLI entry points: `fleetopt-download`, `fleetopt-features`, `fleetopt-train`, `fleetopt-replay`, `fleetopt-experiment`.

Stack: Python 3.11+, DuckDB, pandas, numpy, LightGBM, scikit-learn, OR-Tools (CP-SAT), FastAPI, Pydantic, prometheus-client, pytest, ruff.

---

## 6. Phases

Work one phase at a time. **At the end of each phase: run tests and ruff, commit, and stop to give me a short summary of what was built, the key concepts involved, and what I should read in the code to understand it.** Don't start the next phase until I say so.

### Phase 0: Setup and data exploration
- Scaffold the repo (pyproject, ruff, pytest, Makefile, .gitignore, CI skeleton).
- `download.py` for HVFHV months + zone lookup + shapefile.
- Exploration notebook or script in DuckDB: schema, row counts, nulls, distributions of wait time (`on_scene - request`, `pickup - request`), trip time, fare, demand by hour/day/zone. Save key stats to `reports/eda.json`.
- **Output to me:** what the data looks like, the cleaning rules you propose, and the geofence + sampling rate you recommend (with reasoning).

### Phase 1: Features and models
- `features.py`: one DuckDB `CREATE TABLE trips AS SELECT ... FROM read_parquet(glob, union_by_name = true)` with cleaning filters, plus derived columns: `wait_minutes = on_scene - request` and `pickup_delay = pickup - request` (for simulator calibration), duration, calendar fields, and `fare`.
- Demand table at **zone × 15-minute, bucketed by `request_datetime`**, on a complete grid (zero-demand rows included), restricted to geofence zones. Causal lags via window functions: 15 min, 1 h, 1 day (96), **1 week (672)**, plus rolling means over 1 h and 1 day that exclude the current interval. Train split starts 7 days in so the weekly lag exists.
- `artifacts.py`: `EtaFeatureEncoder` (train-only route/zone medians with fallbacks) and `EtaBundle` / `DemandBundle` dataclasses holding model + encoder + version + metrics, joblib save, type-checked load.
- `travel.py`: zone-to-zone median travel time and distance from the train window, with fallbacks (zone-level, borough-level, distance-based) for sparse pairs. Used by the simulator and both optimizers.
- `train.py`: LightGBM L1 ETA (MAE, RMSE, p90 abs error) on a seeded 2M-row reservoir sample; LightGBM Poisson demand (MAE, RMSE, WMAPE). **Each compared against a naive baseline** (ETA: route median; demand: same-slot-last-week). Train on train, tune on val, report once on test. Write `reports/evaluation.json` (row counts, split dates, val + test metrics, baselines, `limitations` list), save bundles to `artifacts/`.

### Phase 2: Simulator and dispatch (baseline system complete)
- `simulator.py`: time-stepped (e.g. 30 s ticks). Vehicle states: idle, en-route-to-pickup, on-trip, repositioning. Requests come from real test-period HVFHV requests in the geofence (sampled). Travel times from `travel.py`. Requests **abandon** if unassigned after a patience threshold (calibrate from real wait-time distribution). Initial fleet placement proportional to historical demand. Seeded and deterministic.
- `dispatch.py`: (a) greedy nearest-idle-vehicle; (b) **CP-SAT batch assignment** every tick: minimize total pickup time with a max-pickup-time constraint and a penalty for leaving requests unassigned. CP-SAT handles unequal vehicle/request counts directly (no dummy padding) and lets us add side constraints. Optionally keep an idle-time fairness credit as a tunable parameter. Scale costs to integers (CP-SAT needs integer coefficients).
- Also keep a **static batch replay** mode (independent batches, sampled supply) as a quick sanity check, then make the time-stepped simulator the main evaluation.
- Marketplace metrics (define precisely in `metrics.py`): fulfillment rate, mean/p90 passenger wait, utilization (% vehicle-time on trip), deadhead share (empty miles / total miles), idle time, revenue per vehicle-hour.
- `replay.py`: run greedy vs CP-SAT over N test days, write `reports/replay.json`.
- FastAPI via a `create_app(artifact_dir)` factory: artifacts loaded once in `lifespan` (directory from env var), endpoints `/health` (503 if artifacts missing), `/v1/eta`, `/v1/demand`, `/v1/dispatch`, later `/v1/rebalance`; model version in every response; dedicated Prometheus registry with per-endpoint request Counter + latency Histogram at `/metrics`. Latency benchmark script writing `reports/latency.json` (warm-up + 500 requests, in-process, labeled as such).
- Delivery: Makefile targets per stage; Dockerfile (python-slim, `libgomp1`, non-root user); docker-compose; CI running `ruff check`, `ruff format --check`, `pytest --cov`, `docker build`.
- Docs: ARCHITECTURE, MODEL_CARD, DATA_CARD, README with results table generated from reports.
- **Checkpoint:** this is the "baseline is working end to end" milestone.

### Phase 3: Vehicle-value model (enhancement 1)
- Define the target clearly: **value of an idle vehicle in zone z at time t** = expected fare revenue it earns in the next H minutes (e.g. H = 30), net of deadhead cost.
- Two approaches; build (a) first, then (b):
  - (a) Analytic proxy from data: forecast demand × average fare × estimated match probability given current supply.
  - (b) Learned value: run baseline simulations, log every idle event (zone, time, local supply/demand state) and the revenue that vehicle actually earned in the next H minutes, then train LightGBM on those logs (temporal split across sim days).
- Evaluate (b) against (a) on held-out sim days; write `reports/value_model.json`.

### Phase 4: CP-SAT idle-fleet rebalancing (enhancement 2)
- Every K minutes (e.g. 5), take idle vehicles not assigned to requests. Decision `x[i,j] ∈ {0,1}`: vehicle i repositions to zone j (staying put is allowed).
- Objective: maximize Σ x[i,j] · (value_j − λ · reposition_time_ij).
- Constraints: each vehicle at most one move; max reposition time; **diminishing returns** so the solver doesn't send everyone to one zone: cap vehicles sent to zone j by predicted unmet demand (forecast demand − vehicles already there/arriving), or use piecewise-linear marginal value.
- Include a solver time limit and log solve times; fall back to "stay put" if no solution in time.
- Unit-test on tiny hand-checkable instances.

### Phase 5: Offline policy experiment (enhancement 3)
- Policies: **Control** = CP-SAT dispatch, no rebalancing. **Treatment** = CP-SAT dispatch + value-model-driven CP-SAT rebalancing. Optional middle arm: rebalancing driven by the simple demand forecast, to isolate the value model's contribution.
- Design: run every policy on the **same** real demand days with the same seeds (paired design). Report per-metric lift with **bootstrap confidence intervals over days**.
- Trade-off analysis: sweep λ and fleet size; plot utilization vs. wait time vs. deadhead share (Pareto frontier). Explicitly answer "couldn't you raise utilization just by sending cars farther?" with the deadhead data.
- Write `reports/experiment.json` and charts into `reports/figures/`.

### Phase 6: Online experiment design and polish
- `docs/EXPERIMENT_DESIGN.md`: how this would be tested live. Explain **why user-level A/B testing breaks in a marketplace** (interference through shared supply) and propose a **switchback design** (alternate policies by region × time window). Cover primary metric, guardrails (p90 wait, fulfillment, deadhead), power / minimum detectable effect, duration, ramp plan.
- Bonus: simulate a switchback experiment in our simulator and show it recovers the effect measured in Phase 5.
- Final README: problem, architecture, results tables (from reports), honest limitations (simulated fleet, sampled demand, zone-level travel times, no traffic/charging), and a "how to reproduce" section.

---

## 7. How to work with me

- Start in **plan mode**: propose the plan for the current phase, wait for my OK, then implement.
- Small, focused commits per milestone.
- Tests use small synthetic data so they run in seconds without downloads.
- When there's a real design choice (geofence, tick size, H, λ, abandonment threshold), present 2-3 options with trade-offs and a recommendation, and let me decide.
- Explain concepts as we go (CP-SAT modeling, Poisson objectives, point-in-time leakage, paired bootstrap, switchback designs) in plain language. I need to be able to defend every part of this in an interview.
- Never invent results. If something fails or a number looks off, tell me.

---

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
