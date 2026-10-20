"""Point-in-time feature tables in a persistent DuckDB: `trips` and `demand_features`.

`trips` holds every cleaned NYC row plus calendar fields and the `servable` flag. The demand grid
is zone x 15-min slot on the naive local-time axis, so lags are wall-clock aligned. Slots that do
not exist (DST spring-forward) keep a row with a NULL count and are excluded from every split.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb

from fleetopt import constants as C
from fleetopt import eda, fare_anomaly, zones
from fleetopt.db import connect


def dst_gap_slots(start: str, end: str) -> list[datetime]:
    """Naive 15-min slot starts in [start, end) that do not exist in local time."""
    tz, utc = ZoneInfo(C.LOCAL_TZ), ZoneInfo("UTC")
    t, stop, gap = datetime.fromisoformat(start), datetime.fromisoformat(end), []
    while t < stop:
        if t.replace(tzinfo=tz).astimezone(utc).astimezone(tz).replace(tzinfo=None) != t:
            gap.append(t)
        t += timedelta(minutes=C.SLOT_MINUTES)
    return gap


def _holiday_list() -> str:
    return ", ".join(f"DATE '{d}'" for d in C.HOLIDAYS)


def build_trips(
    con: duckdb.DuckDBPyConnection,
    raw_dir: Path = C.RAW_DIR,
    months: tuple[str, ...] | None = None,
    thin_percent: int | None = None,
    until: str | None = None,
) -> None:
    """Table `trips` from the shared cleaning rules (eda.build_base/build_clean)."""
    eda.build_base(con, raw_dir, months)
    eda.build_clean(con)
    ids = ", ".join(str(i) for i in zones.geofence_zone_ids())
    where = ["TRUE"]
    if thin_percent:
        where.append(f"hash(request_datetime, pu, do_, hvfhs_license_num) % 100 < {thin_percent}")
    if until:
        where.append(f"request_datetime < TIMESTAMP '{until}'")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE trips AS
        SELECT month, hvfhs_license_num AS license, request_datetime, on_scene_datetime,
               pickup_datetime, dropoff_datetime, pu, do_, trip_miles,
               trip_time / 60.0 AS duration_min, fare, fare_invalid, fare_raw,
               wait_min, pickup_delay_min, wait_invalid, driver_pay,
               shared_match_flag, wav_request_flag,
               hour(request_datetime) AS hour,
               isodow(request_datetime) - 1 AS dow,
               (hour(request_datetime) * 60 + minute(request_datetime)) // {C.SLOT_MINUTES}
                 AS slot_of_day,
               isodow(request_datetime) >= 6 AS is_weekend,
               CAST(request_datetime AS DATE) IN ({_holiday_list()}) AS us_holiday,
               (pu IN ({ids}) AND do_ IN ({ids})) AS servable
        FROM clean WHERE {" AND ".join(where)}
        """
    )


def build_demand_features(
    con: duckdb.DuckDBPyConnection,
    splits: dict[str, tuple[str, str]],
    grid: tuple[str, str],
    horizon: int = C.DEMAND_HORIZON,
) -> None:
    """Table `demand_features`: causal lag/rolling features per zone x slot for `horizon`.

    For horizon h the newest observation any feature may read is slot t-h: lags k < h are not
    produced and rolling means end at t-h.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    ids = zones.geofence_zone_ids()
    con.execute("CREATE OR REPLACE TABLE geofence_zones AS SELECT unnest($1) AS zone", [ids])
    gap = dst_gap_slots(*grid)
    con.execute("CREATE OR REPLACE TABLE dst_gap(slot_ts TIMESTAMP)")
    if gap:
        con.executemany("INSERT INTO dst_gap VALUES (?)", [[g] for g in gap])
    con.execute(
        f"""
        CREATE OR REPLACE TABLE demand AS
        WITH slots AS (
          SELECT unnest(generate_series(TIMESTAMP '{grid[0]}',
                 TIMESTAMP '{grid[1]}' - INTERVAL {C.SLOT_MINUTES} MINUTE,
                 INTERVAL {C.SLOT_MINUTES} MINUTE)) AS slot_ts
        ), counts AS (
          SELECT pu AS zone,
                 time_bucket(INTERVAL {C.SLOT_MINUTES} MINUTE, request_datetime) AS slot_ts,
                 count(*) AS n
          FROM trips WHERE servable GROUP BY ALL
        )
        SELECT z.zone, s.slot_ts,
               CASE WHEN g.slot_ts IS NOT NULL THEN NULL ELSE coalesce(c.n, 0) END AS count,
               g.slot_ts IS NOT NULL AS is_dst_gap
        FROM geofence_zones z CROSS JOIN slots s
        LEFT JOIN counts c ON c.zone = z.zone AND c.slot_ts = s.slot_ts
        LEFT JOIN dst_gap g ON g.slot_ts = s.slot_ts
        """
    )
    split_sql = (
        "CASE "
        + " ".join(
            f"WHEN slot_ts >= TIMESTAMP '{a}' AND slot_ts < TIMESTAMP '{b}' THEN '{name}'"
            for name, (a, b) in splits.items()
        )
        + " END"
    )
    w = "PARTITION BY zone ORDER BY slot_ts"
    lags = ", ".join(
        f"lag(count, {k}) OVER ({w}) AS lag_{k}" for k in C.DEMAND_LAGS if k >= horizon
    )
    rolls = ", ".join(
        f"avg(count) OVER ({w} ROWS BETWEEN {n + horizon - 1} PRECEDING AND {horizon} PRECEDING)"
        f" AS roll_mean_{n}"
        for n in C.DEMAND_ROLLING
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE demand_stage AS
        SELECT zone, slot_ts, count, is_dst_gap,
               CASE WHEN is_dst_gap THEN NULL ELSE {split_sql} END AS split,
               {lags}, {rolls},
               (hour(slot_ts) * 60 + minute(slot_ts)) // {C.SLOT_MINUTES} AS slot_of_day,
               isodow(slot_ts) - 1 AS dow, isodow(slot_ts) >= 6 AS is_weekend,
               CAST(slot_ts AS DATE) IN ({_holiday_list()}) AS us_holiday
        FROM demand
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE demand_features AS
        SELECT s.*, r.zone_mean_rate, {horizon} AS horizon
        FROM demand_stage s
        LEFT JOIN (SELECT zone, avg(count) AS zone_mean_rate FROM demand_stage
                   WHERE split = 'train' GROUP BY zone) r USING (zone)
        """
    )
    con.execute("DROP TABLE demand_stage")


def run(dev: bool = False, raw_dir: Path = C.RAW_DIR, reports_dir: Path = C.REPORTS_DIR) -> dict:
    con = connect(C.DEV_DUCKDB_PATH if dev else C.DUCKDB_PATH)
    if dev:
        build_trips(con, raw_dir, C.DEV_MONTHS, C.DEV_THIN_PERCENT, C.DEV_GRID_BOUNDS[1])
    else:
        build_trips(con, raw_dir)
    build_demand_features(
        con, C.DEV_SPLITS if dev else C.SPLITS, C.DEV_GRID_BOUNDS if dev else C.GRID_BOUNDS
    )
    if not dev:
        fare_anomaly.run(con, reports_dir)
    con.execute("DROP VIEW IF EXISTS clean")
    con.execute("DROP TABLE IF EXISTS base")
    summary = {
        "trips": con.execute("SELECT count(*) FROM trips").fetchone()[0],
        "demand_rows": con.execute("SELECT count(*) FROM demand_features").fetchone()[0],
        "dst_gap_rows": con.execute(
            "SELECT count(*) FROM demand_features WHERE is_dst_gap"
        ).fetchone()[0],
        "split_rows": dict(
            con.execute(
                "SELECT split, count(*) FROM demand_features WHERE split IS NOT NULL GROUP BY 1"
            ).fetchall()
        ),
    }
    con.close()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", action="store_true", help="Jan 1-21, 1% of trips, separate DB")
    print(run(dev=ap.parse_args().dev))


if __name__ == "__main__":
    main()
