"""DuckDB exploration of raw HVFHV files -> reports/eda.json and reports/data_quality.json.

Everything heavy runs in DuckDB; only small aggregates reach Python.
"""

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from fleetopt import constants as C
from fleetopt.cleaning import FARE_INVALID, RULES, WAIT_ANOMALIES
from fleetopt.db import connect

GEOFENCE_NEAR_KM = 4.0  # "inner" Brooklyn/Queens = centroid within this of a Manhattan centroid
LAT_96TH_ST = 40.7855
TARGET_UTILIZATION = 0.6
LICENSES = {"HV0002": "Juno", "HV0003": "Uber", "HV0004": "Via", "HV0005": "Lyft"}


def _glob(raw_dir: Path, months: tuple[str, ...] | None = None) -> str:
    if months is None:
        return str(raw_dir / "fhvhv_tripdata_*.parquet").replace("\\", "/")
    # DuckDB accepts a list literal; callers embed it unquoted via _sources().
    return ", ".join(f"'{(raw_dir / f'fhvhv_tripdata_{m}.parquet').as_posix()}'" for m in months)


def _sources(raw_dir: Path, months: tuple[str, ...] | None) -> str:
    return f"[{_glob(raw_dir, months)}]" if months else f"'{_glob(raw_dir)}'"


def build_base(
    con: duckdb.DuckDBPyConnection, raw_dir: Path, months: tuple[str, ...] | None = None
) -> None:
    """Table `base`: raw rows plus keep-flags per drop rule, wait_invalid, and derived minutes."""
    keep = ", ".join(f"({cond}) AS ok_{name.replace('.', '_')}" for name, cond in RULES.items())
    anomalies = ", ".join(
        f"coalesce({cond}, false) AS an_{n}" for n, cond in WAIT_ANOMALIES.items()
    )
    any_anomaly = " OR ".join(f"an_{n}" for n in WAIT_ANOMALIES)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE base AS
        WITH raw AS (
          SELECT *, strptime(regexp_extract(filename, '(\\d{{4}}-\\d{{2}})', 1) || '-01', '%Y-%m-%d')
                    AS month_start
          FROM read_parquet({_sources(raw_dir, months)}, filename = true, union_by_name = true)
        ), m AS (
          SELECT *, month_start + INTERVAL 1 MONTH AS month_end FROM raw
        ), f AS (
          SELECT *, {anomalies}, {keep}, ({FARE_INVALID}) AS fare_invalid FROM m
        )
        SELECT strftime(month_start, '%Y-%m') AS month, hvfhs_license_num, request_datetime,
               on_scene_datetime, pickup_datetime, dropoff_datetime,
               PULocationID AS pu, DOLocationID AS do_, trip_miles, trip_time,
               CASE WHEN fare_invalid THEN NULL ELSE base_passenger_fare END AS fare, fare_invalid,
               base_passenger_fare AS fare_raw,
               driver_pay, shared_request_flag, shared_match_flag,
               wav_request_flag, access_a_ride_flag,
               ({any_anomaly}) AS wait_invalid,
               CASE WHEN NOT ({any_anomaly})
                    THEN date_diff('second', request_datetime, on_scene_datetime) / 60.0 END
                 AS wait_min,
               CASE WHEN NOT ({any_anomaly})
                    THEN date_diff('second', request_datetime, pickup_datetime) / 60.0 END
                 AS pickup_delay_min,
               {", ".join(f"an_{n}" for n in WAIT_ANOMALIES)},
               {", ".join("ok_" + n.replace(".", "_") for n in RULES)}
        FROM f
        """
    )


def _q(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict]:
    df = con.execute(sql).df()
    return json.loads(df.to_json(orient="records"))


def data_quality(con: duckdb.DuckDBPyConnection) -> dict:
    total = con.execute("SELECT count(*) FROM base").fetchone()[0]
    names = [f"ok_{n.replace('.', '_')}" for n in RULES]
    per_rule = {}
    for rule, col in zip(RULES, names, strict=True):
        # NULL comparisons count as failures: coalesce to FALSE.
        dropped = con.execute(f"SELECT count(*) FROM base WHERE NOT coalesce({col}, false)")
        per_rule[rule] = dropped.fetchone()[0]
    waterfall, alive = [], "TRUE"
    prev = total
    for rule, col in zip(RULES, names, strict=True):
        alive += f" AND coalesce({col}, false)"
        remaining = con.execute(f"SELECT count(*) FROM base WHERE {alive}").fetchone()[0]
        waterfall.append({"rule": rule, "dropped_here": prev - remaining, "remaining": remaining})
        prev = remaining
    return {
        "raw_rows": total,
        "clean_rows": prev,
        "dropped_total": total - prev,
        "dropped_share": (total - prev) / total,
        "per_rule_independent_drops": per_rule,
        "sequential_waterfall": waterfall,
        "wait_fields_nulled": wait_flag_counts(con, "clean"),
        "fare_nulled": _q(
            con,
            """SELECT count(*) FILTER (WHERE fare_invalid) AS fare_invalid, count(*) AS rows,
                      count(*) FILTER (WHERE fare_invalid) * 1.0 / count(*) AS share FROM clean""",
        )[0],
        "note": "Independent drops overlap; waterfall drops sum to total. Wait anomalies and invalid "
        "fares keep the row but null the field (wait_invalid / fare_invalid = true).",
    }


def wait_flag_counts(con: duckdb.DuckDBPyConnection, table: str) -> dict:
    cols = ", ".join(f"count(*) FILTER (WHERE an_{n}) AS {n}" for n in WAIT_ANOMALIES)
    row = _q(
        con,
        f"SELECT count(*) AS rows, count(*) FILTER (WHERE wait_invalid) AS wait_invalid, {cols} "
        f"FROM {table}",
    )[0]
    row["wait_invalid_share"] = row["wait_invalid"] / row["rows"]
    return row


def build_clean(con: duckdb.DuckDBPyConnection) -> None:
    ok = " AND ".join(f"coalesce(ok_{n.replace('.', '_')}, false)" for n in RULES)
    con.execute(f"CREATE OR REPLACE VIEW clean AS SELECT * FROM base WHERE {ok}")


def distributions(con: duckdb.DuckDBPyConnection) -> dict:
    def qs(col: str, where: str = "TRUE") -> dict:
        row = con.execute(
            f"""SELECT count({col}) n, avg({col}) mean, min({col}) min,
                   approx_quantile({col}, [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]) q, max({col}) max
                FROM clean WHERE {where}"""
        ).fetchone()
        keys = ["p10", "p25", "p50", "p75", "p90", "p99"]
        return {
            "n": row[0],
            "mean": row[1],
            "min": row[2],
            **dict(zip(keys, row[3], strict=True)),
            "max": row[4],
        }

    return {
        "wait_min_on_scene_minus_request": qs("wait_min"),
        "pickup_delay_min_pickup_minus_request": qs("pickup_delay_min"),
        "trip_time_min": qs("trip_time / 60.0"),
        "trip_miles": qs("trip_miles"),
        "base_passenger_fare": qs("fare"),
        "pickup_delay_min_uber_only": qs("pickup_delay_min", "hvfhs_license_num = 'HV0003'"),
    }


def license_breakdown(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Share of rows and null rates by license (checks whether on_scene is Uber-only)."""
    return _q(
        con,
        """
        SELECT hvfhs_license_num AS license, count(*) AS rows,
               count(*) * 1.0 / sum(count(*)) OVER () AS share,
               avg((on_scene_datetime IS NULL)::int) AS on_scene_null_rate,
               avg((request_datetime IS NULL)::int) AS request_null_rate,
               avg((shared_match_flag = 'Y')::int) AS shared_match_share,
               avg((wav_request_flag = 'Y')::int) AS wav_request_share,
               avg((access_a_ride_flag = 'Y')::int) AS access_a_ride_share
        FROM base GROUP BY 1 ORDER BY rows DESC
        """,
    )


def null_rates_all(con: duckdb.DuckDBPyConnection) -> dict:
    cols = [r[0] for r in con.execute("DESCRIBE base").fetchall() if not r[0].startswith("ok_")]
    exprs = ", ".join(f"avg(({c} IS NULL)::int) AS {c}" for c in cols)
    return _q(con, f"SELECT {exprs} FROM base")[0]


def demand_profile(con: duckdb.DuckDBPyConnection) -> dict:
    # Demand is bucketed by REQUEST time (the true demand signal), not pickup time.
    return {
        "by_hour_of_day": _q(
            con,
            """SELECT hour(request_datetime) AS hour, count(*) AS requests,
                      count(*) * 1.0 / count(DISTINCT date_trunc('day', request_datetime)) AS per_day
               FROM clean GROUP BY 1 ORDER BY 1""",
        ),
        "by_day_of_week": _q(
            con,
            """SELECT isodow(request_datetime) AS iso_dow, count(*) AS requests
               FROM clean GROUP BY 1 ORDER BY 1""",
        ),
        "top_pickup_zones": _q(
            con,
            f"""SELECT c.pu AS location_id, z.zone, z.borough, count(*) AS requests
                FROM clean c JOIN '{C.ZONES_PARQUET.as_posix()}' z ON z.location_id = c.pu
                GROUP BY ALL ORDER BY requests DESC LIMIT 15""",
        ),
        "by_borough": _q(
            con,
            f"""SELECT z.borough, count(*) AS requests, count(*) * 1.0 / sum(count(*)) OVER () AS share
                FROM clean c JOIN '{C.ZONES_PARQUET.as_posix()}' z ON z.location_id = c.pu
                GROUP BY 1 ORDER BY requests DESC""",
        ),
    }


def geofence_options(con: duckdb.DuckDBPyConnection) -> list[dict]:
    zones = con.execute(
        f"SELECT location_id, borough, lat, x_ft, y_ft FROM '{C.ZONES_PARQUET.as_posix()}' "
        "WHERE location_id BETWEEN 1 AND 263 AND lat IS NOT NULL"
    ).df()
    man = zones[zones.borough == "Manhattan"]
    near_ft = GEOFENCE_NEAR_KM * 1000 / 0.3048
    inner = zones[
        zones.borough.isin(["Brooklyn", "Queens"])
        & zones.apply(
            lambda r: (
                (((man.x_ft - r.x_ft) ** 2 + (man.y_ft - r.y_ft) ** 2) ** 0.5).min() <= near_ft
            ),
            axis=1,
        )
    ]
    options = {
        "manhattan": sorted(man.location_id),
        "manhattan_below_96th": sorted(man[man.lat < LAT_96TH_ST].location_id),
        f"manhattan_plus_inner_bk_qns_{GEOFENCE_NEAR_KM:g}km": sorted(
            list(man.location_id) + list(inner.location_id)
        ),
    }
    n_days = con.execute("SELECT count(DISTINCT date_trunc('day', request_datetime)) FROM clean")
    n_days = n_days.fetchone()[0]
    out = []
    for name, ids in options.items():
        gf_ids = pd.DataFrame({"id": ids})  # noqa: F841 (read by DuckDB replacement scan)
        con.execute("CREATE OR REPLACE TEMP TABLE gf AS SELECT id FROM gf_ids")
        # Intra-geofence trips: both ends inside; those are the requests a closed fleet can serve.
        s = con.execute(
            """
            WITH t AS (
              SELECT c.*, (c.do_ IN (SELECT id FROM gf)) AS intra
              FROM clean c WHERE c.pu IN (SELECT id FROM gf)
            ), hourly AS (
              SELECT dayofweek(request_datetime) d, hour(request_datetime) h,
                     count(*) FILTER (WHERE intra) * 1.0
                       / count(DISTINCT date_trunc('day', request_datetime)) AS reqs_per_hour
              FROM t GROUP BY 1, 2
            )
            SELECT (SELECT count(*) FROM t), (SELECT avg(intra::int) FROM t),
                   (SELECT avg(trip_time) / 60 FROM t WHERE intra),
                   (SELECT avg(trip_miles) FROM t WHERE intra),
                   (SELECT avg(pickup_delay_min) FROM t WHERE intra),
                   (SELECT max(reqs_per_hour) FROM hourly), (SELECT avg(reqs_per_hour) FROM hourly)
            """
        ).fetchone()
        n_pickups, intra_share, trip_min, trip_mi, pickup_min, peak_hr, mean_hr = s
        sizing = {}
        for rate in (0.05, 0.10):
            # Little's law: busy vehicles = arrival rate x time on trip; fleet = busy / target util.
            peak_busy = peak_hr * rate * trip_min / 60
            sizing[f"sample_{rate:.0%}"] = {
                "peak_requests_per_hour": peak_hr * rate,
                "mean_requests_per_hour": mean_hr * rate,
                "fleet_for_peak_at_target_util": peak_busy / TARGET_UTILIZATION,
                "fleet_for_mean_at_target_util": mean_hr
                * rate
                * trip_min
                / 60
                / TARGET_UTILIZATION,
            }
        out.append(
            {
                "name": name,
                "n_zones": len(ids),
                "zone_ids": ids,
                "pickups_per_day": n_pickups / n_days,
                "intra_geofence_share": intra_share,
                "intra_mean_trip_min": trip_min,
                "intra_mean_trip_miles": trip_mi,
                "intra_mean_pickup_delay_min": pickup_min,
                "little_law_fleet_sizing": sizing,
                "target_utilization": TARGET_UTILIZATION,
            }
        )
    return out


# Thresholds for flagging a month as materially different from the first (baseline) month.
MATERIAL = {
    "license_share_abs": 0.02,
    "on_scene_null_rate_abs": 0.02,
    "wait_quantile_rel": 0.10,
    "drop_share_rel": 0.25,
    "wait_invalid_share_rel": 0.25,
}


def month_comparison(con: duckdb.DuckDBPyConnection) -> dict:
    """Per-month license mix, on_scene nulls, wait percentiles, drops; flags vs the first month."""
    months = [r[0] for r in con.execute("SELECT DISTINCT month FROM base ORDER BY 1").fetchall()]
    out: dict[str, dict] = {}
    for m in months:
        raw = con.execute(f"SELECT count(*) FROM base WHERE month = '{m}'").fetchone()[0]
        clean = con.execute(f"SELECT count(*) FROM clean WHERE month = '{m}'").fetchone()[0]
        lic = {
            r["license"]: r
            for r in _q(
                con,
                f"""SELECT hvfhs_license_num AS license, count(*) * 1.0 / {raw} AS share,
                       avg((on_scene_datetime IS NULL)::int) AS on_scene_null_rate
                    FROM base WHERE month = '{m}' GROUP BY 1""",
            )
        }
        wq = con.execute(
            f"""SELECT approx_quantile(wait_min, [0.5, 0.9, 0.99]) FROM clean
                WHERE month = '{m}' AND hvfhs_license_num = 'HV0003'"""
        ).fetchone()[0]
        pq = con.execute(
            f"""SELECT approx_quantile(pickup_delay_min, [0.5, 0.9, 0.99]) FROM clean
                WHERE month = '{m}'"""
        ).fetchone()[0]
        drops = {
            rule: con.execute(
                f"SELECT count(*) FROM base WHERE month = '{m}' "
                f"AND NOT coalesce(ok_{rule.replace('.', '_')}, false)"
            ).fetchone()[0]
            for rule in RULES
        }
        wf = _q(
            con,
            f"""SELECT count(*) FILTER (WHERE wait_invalid) * 1.0 / count(*) AS share
                FROM clean WHERE month = '{m}'""",
        )[0]
        out[m] = {
            "raw_rows": raw,
            "clean_rows": clean,
            "dropped_share": 1 - clean / raw,
            "license_share": {k: v["share"] for k, v in lic.items()},
            "on_scene_null_rate_by_license": {k: v["on_scene_null_rate"] for k, v in lic.items()},
            "uber_wait_min_p50_p90_p99": wq,
            "pickup_delay_min_p50_p90_p99": pq,
            "independent_drops": drops,
            "wait_invalid_share_of_clean": wf["share"],
            "fare_invalid_share_of_clean": con.execute(
                f"SELECT avg(fare_invalid::int) FROM clean WHERE month = '{m}'"
            ).fetchone()[0],
        }
    base_m = months[0]
    b = out[base_m]
    flags: dict[str, list[str]] = {m: [] for m in months[1:]}
    for m in months[1:]:
        c = out[m]
        for lic in set(b["license_share"]) | set(c["license_share"]):
            d = abs(c["license_share"].get(lic, 0) - b["license_share"].get(lic, 0))
            if d > MATERIAL["license_share_abs"]:
                flags[m].append(
                    f"license share {lic}: {b['license_share'].get(lic, 0):.3f} -> "
                    f"{c['license_share'].get(lic, 0):.3f}"
                )
        for lic, v in c["on_scene_null_rate_by_license"].items():
            d = abs(v - b["on_scene_null_rate_by_license"].get(lic, v))
            if d > MATERIAL["on_scene_null_rate_abs"]:
                flags[m].append(
                    f"on_scene null rate {lic}: "
                    f"{b['on_scene_null_rate_by_license'].get(lic):.3f} -> {v:.3f}"
                )
        for key in ("uber_wait_min_p50_p90_p99", "pickup_delay_min_p50_p90_p99"):
            for name, x, y in zip(("p50", "p90", "p99"), b[key], c[key], strict=True):
                if abs(y - x) / x > MATERIAL["wait_quantile_rel"]:
                    flags[m].append(f"{key} {name}: {x:.2f} -> {y:.2f}")
        if (
            abs(c["dropped_share"] - b["dropped_share"]) / b["dropped_share"]
            > MATERIAL["drop_share_rel"]
        ):
            flags[m].append(f"drop share: {b['dropped_share']:.4f} -> {c['dropped_share']:.4f}")
        for rule in RULES:
            bx, cx = (
                b["independent_drops"][rule] / b["raw_rows"],
                c["independent_drops"][rule] / c["raw_rows"],
            )
            if bx > 0.001 and abs(cx - bx) / bx > MATERIAL["drop_share_rel"]:
                flags[m].append(f"rule {rule} drop share: {bx:.4f} -> {cx:.4f}")
        bw, cw = b["wait_invalid_share_of_clean"], c["wait_invalid_share_of_clean"]
        if abs(cw - bw) / bw > MATERIAL["wait_invalid_share_rel"]:
            flags[m].append(f"wait_invalid share: {bw:.4f} -> {cw:.4f}")
    return {"baseline_month": base_m, "thresholds": MATERIAL, "months": out, "flags": flags}


def run(raw_dir: Path = C.RAW_DIR, reports_dir: Path = C.REPORTS_DIR) -> dict:
    con = connect()
    build_base(con, raw_dir)
    build_clean(con)
    dq = data_quality(con)
    eda = {
        "files": sorted(p.name for p in raw_dir.glob("fhvhv_tripdata_*.parquet")),
        "license_breakdown": license_breakdown(con),
        "null_rates_raw": null_rates_all(con),
        "distributions_clean": distributions(con),
        "demand_profile": demand_profile(con),
        "geofence_options": geofence_options(con),
        "month_comparison": month_comparison(con),
        "license_names": LICENSES,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "data_quality.json").write_text(json.dumps(dq, indent=2))
    (reports_dir / "eda.json").write_text(json.dumps(eda, indent=2))
    return {"data_quality": dq, "eda": eda}


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
    print(f"Wrote {C.REPORTS_DIR / 'eda.json'} and {C.REPORTS_DIR / 'data_quality.json'}")


if __name__ == "__main__":
    main()
