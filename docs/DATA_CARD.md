# Data card (first draft, Phase 1)

## Source
NYC TLC High Volume For-Hire Vehicle (HVFHV) trip records, Jan-Apr 2025 (80.0M raw rows), plus
the TLC taxi zone lookup and zone shapefile (centroids). Licenses: HV0003 Uber, HV0005 Lyft
(Juno/Via negligible). Timestamps are **naive local time**.

Splits (request time, [start, end)): train 2025-01-08..03-01, validation March, test April
minus its last 30 minutes. January 1-7 only feeds lag features.

## Cleaning (`src/zone_ranger/cleaning.py`, shared by `eda.py` and `features.py`)
* **Dropped (4.29% of raw, 76.6M rows kept):** invalid zones (outside 1-263), trip time outside
  1-180 min, trip miles outside 0.1-100, missing request time, pickup outside its file month,
  dropoff not after pickup. Per-rule counts: `reports/data_quality.json`.
* **Flagged, row kept - waits:** request after pickup, on-scene before request, or pickup delay
  over 3 h null the wait fields and set `wait_invalid` (1.29% of clean rows).
* **Flagged, row kept - fares:** fare <= 0 or >= 500 nulls `fare` and sets `fare_invalid`
  (91,627 rows, 0.12%). Fare-based work (value model, revenue metrics) must exclude flagged rows.
  `fare_raw` keeps the original value. Diagnosis: `reports/fare_anomaly.json`.

## Wait fields
Waits (`wait_min`, `pickup_delay_min`) exist for **completed trips only**; cancelled or abandoned
requests are not in the data. Any patience / abandonment threshold in the simulator is therefore
an assumption informed by the wait tail, not a measurement. `on_scene_datetime` is missing for
Lyft before 2025-03-06, so wait calibration is **Uber-only** in all months.

## Service area and demand
Geofence = Manhattan, 69 zones. A request is `servable` when pickup AND dropoff are inside it
(about 72.7% of Manhattan pickups). Demand = servable requests per pickup zone per 15-min slot,
bucketed by `request_datetime` (full, unsampled; sampling applies only inside the simulator).
Zero-demand cells are kept (complete zone x slot grid).

## Time handling
* **DST:** timestamps are naive local time, so 2025-03-09 02:00-02:59 does not exist (a handful
  of rows land there). The grid keeps those 4 x 69 cells with `count = NULL`, `is_dst_gap = TRUE`;
  they belong to no split and window functions skip them. There is no fall-back hour in range.
* **Right-censoring:** trips live in the file of their *pickup* month, so April's file lacks trips
  requested late on Apr 30 but picked up in May. The test window stops at 2025-04-30 23:30.
  Month boundaries Jan/Feb/Mar are continuous because bucketing is by request time.

## Fare anomaly (decision 8)
Fare <= 0 rows by month: Jan 16,261; Feb 1,690; **Mar 60,919**; Apr 12,740 (clean rows only).
The March excess is almost entirely Lyft (59,491 of 60,919) with exactly zero recorded fare while
`driver_pay` is positive (60,893 of 60,919), concentrated on a few days (Mar 16, 21, 22, 26
account for ~58k). Same signature on Jan 21 and Apr 24. These look like missing fare capture on
specific days, not free rides; the rows are still real trips for demand and ETA.

## Known limitations
Demand outside the geofence is not modelled; pickup delay includes dispatcher-to-driver time that
the data does not separate; shared rides are not specially handled; no weather or event data.
