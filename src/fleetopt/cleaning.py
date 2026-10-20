"""Cleaning rules shared by eda.py and features.py (single source of truth).

Three kinds of rule (decisions 4 and 8 in PROJECT_PLAN.md):
* RULES: row is dropped when the condition is false.
* WAIT_ANOMALIES: row is kept; wait fields are nulled and ``wait_invalid`` is set.
* FARE_INVALID: row is kept; fare is nulled and ``fare_invalid`` is set. Fare-based work
  (value model, revenue metrics) must exclude flagged rows.
"""

# SQL conditions over the raw HVFHV columns plus ``month_start``/``month_end`` (file month).
RULES: dict[str, str] = {
    "valid_zones": "PULocationID BETWEEN 1 AND 263 AND DOLocationID BETWEEN 1 AND 263",
    "trip_time_1_to_180_min": "trip_time BETWEEN 60 AND 10800",
    "trip_miles_0.1_to_100": "trip_miles BETWEEN 0.1 AND 100",
    "request_present": "request_datetime IS NOT NULL",
    "pickup_in_file_month": "pickup_datetime >= month_start AND pickup_datetime < month_end",
    "dropoff_after_pickup": "dropoff_datetime > pickup_datetime",
}

WAIT_ANOMALIES: dict[str, str] = {
    "request_after_pickup": "request_datetime > pickup_datetime",
    "on_scene_before_request": "on_scene_datetime < request_datetime",
    "pickup_delay_gt_3h": "pickup_datetime - request_datetime > INTERVAL 3 HOUR",
}

FARE_INVALID = (
    "base_passenger_fare IS NULL OR base_passenger_fare <= 0 OR base_passenger_fare >= 500"
)


def keep_flag_name(rule: str) -> str:
    return "ok_" + rule.replace(".", "_")
