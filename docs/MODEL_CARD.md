# Model card (first draft, Phase 1)

Numbers are from `reports/evaluation.json`: test = April 2025 (minus last 30 min), model trained on
Jan 8 - Feb 28 only (no refit on val). Hyper-parameters chosen on March; test scored once.

## ETA model (trip duration in minutes)
* **Task / data:** intra-Manhattan trips; 2M-row seeded train sample, full val (5.15M) and test (5.15M).
* **Model:** LightGBM, L1 objective. Best: num_leaves 255, min_data_in_leaf 100, 281 rounds.
* **Features (request-time only):** pickup/dropoff zone, hour, slot of day, dow, weekend, holiday,
  centroid distance (km), train-only median duration for route, pickup zone, dropoff zone and
  route x time band (fallback: route-band -> route -> pickup zone -> global, min 20 trips).
* **Leakage ablation:** adding `trip_miles` (known only after the trip) gives MAE 2.57 vs 3.26.
  That gap is leakage, not a deployable gain; the main model excludes it. The ablation stopped at
  the 600-round cap, so its number is slightly pessimistic for that variant.

| Test | MAE | RMSE | p90 abs err |
|---|---|---|---|
| LightGBM (main) | 3.26 | 5.10 | 6.85 |
| Route median baseline | 3.88 | 5.87 | 8.32 |
| Global median | 6.37 | 8.76 | 13.57 |
| Ablation (+trip_miles) | 2.57 | 4.06 | 5.70 |

Error grows with trip length (MAE short 1.98, medium 2.36, long 5.06 min).

## Demand model (servable requests per zone per 15 min, horizon 1)
* **Model:** LightGBM Poisson. Best: num_leaves 127, lr 0.1, 680 rounds.
* **Features:** lags 1/4/96/672 slots, rolling means over the previous 4/96 slots (all end at
  t - horizon), slot of day, dow, weekend, holiday, zone, train-only zone mean rate.
  Perturbing counts at slots >= t leaves every feature at t unchanged (tested).

| Test | WMAPE | MAE | RMSE | Poisson dev. | Pred/actual total |
|---|---|---|---|---|---|
| LightGBM | 0.160 | 4.15 | 6.36 | 1.30 | 1.006 |
| Same slot last week | 0.239 | 6.21 | 9.89 | 3.87 | 0.985 |
| Same slot yesterday | 0.319 | 8.27 | 14.27 | 6.58 | 0.996 |

WMAPE by zone-volume tier: low 0.249, mid 0.167, high 0.139.

## Travel matrix
Train-window median minutes/miles per zone pair and 4 time bands. Mid-day coverage: 79.3% of cells
from the pair-band median, 5.9% pair all-day, 10.4% pickup-zone, 4.3% haversine fallback
(citywide median speed 9.2 mph).

## Limitations
See the `limitations` list in `reports/evaluation.json`: no refit on train+val, servable-only
demand, Uber-only completed-trip waits, bands ignore weekday/weekend, in-sample route medians for
ETA training rows, DST gap and censoring handling, ablation is not deployable. Test is a single
April window; no weather or event features; one-step-ahead demand only.
