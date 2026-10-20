"""Diagnostic for fare <= 0 rows (decision 8): what explains the March spike?

Reports measured breakdowns by month, day, license and driver_pay status. Nothing is assumed.
"""

import json
from pathlib import Path

import duckdb

from fleetopt import constants as C

_PAY = (
    "CASE WHEN driver_pay IS NULL THEN 'null' WHEN driver_pay = 0 THEN 'zero' ELSE 'positive' END"
)


def _rows(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict]:
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


def diagnose(con: duckdb.DuckDBPyConnection, top_days: int = 15) -> dict:
    bad = "fare_raw <= 0"
    return {
        "by_month": _rows(
            con,
            f"""SELECT month, count(*) AS rows, count(*) FILTER (WHERE {bad}) AS fare_le_0,
                       count(*) FILTER (WHERE {bad}) * 1.0 / count(*) AS share
                FROM trips GROUP BY month ORDER BY month""",
        ),
        "top_days": _rows(
            con,
            f"""SELECT CAST(pickup_datetime AS DATE) AS day, count(*) AS fare_le_0
                FROM trips WHERE {bad} GROUP BY 1 ORDER BY fare_le_0 DESC LIMIT {top_days}""",
        ),
        "by_month_license": _rows(
            con,
            f"""SELECT month, license, count(*) AS fare_le_0 FROM trips WHERE {bad}
                GROUP BY ALL ORDER BY month, fare_le_0 DESC""",
        ),
        "by_month_driver_pay": _rows(
            con,
            f"""SELECT month, {_PAY} AS driver_pay_status, count(*) AS fare_le_0 FROM trips
                WHERE {bad} GROUP BY ALL ORDER BY month, fare_le_0 DESC""",
        ),
        "by_month_zero_vs_negative": _rows(
            con,
            f"""SELECT month, CASE WHEN fare_raw = 0 THEN 'zero' ELSE 'negative' END AS kind,
                       count(*) AS fare_le_0 FROM trips WHERE {bad} GROUP BY ALL
                ORDER BY month, kind""",
        ),
        "other_invalid": _rows(
            con,
            """SELECT count(*) FILTER (WHERE fare_raw IS NULL) AS fare_null,
                      count(*) FILTER (WHERE fare_raw >= 500) AS fare_ge_500 FROM trips""",
        )[0],
    }


def run(con: duckdb.DuckDBPyConnection, reports_dir: Path = C.REPORTS_DIR) -> dict:
    out = diagnose(con)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "fare_anomaly.json").write_text(json.dumps(out, indent=2, default=str))
    return out
