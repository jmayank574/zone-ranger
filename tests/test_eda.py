import json
from datetime import datetime, timedelta

import pandas as pd

from fleetopt import eda
from fleetopt.db import connect


def _rows():
    t0 = datetime(2025, 1, 10, 12)
    base = dict(
        hvfhs_license_num="HV0003",
        dispatching_base_num="B",
        originating_base_num="B",
        trip_miles=2.0,
        trip_time=600,
        base_passenger_fare=15.0,
        tolls=0.0,
        bcf=0.0,
        sales_tax=0.0,
        congestion_surcharge=0.0,
        airport_fee=0.0,
        tips=0.0,
        driver_pay=10.0,
        shared_request_flag="N",
        shared_match_flag="N",
        access_a_ride_flag="N",
        wav_request_flag="N",
        wav_match_flag="N",
        cbd_congestion_fee=0.0,
        PULocationID=4,
        DOLocationID=5,
        request_datetime=t0,
        on_scene_datetime=t0 + timedelta(minutes=3),
        pickup_datetime=t0 + timedelta(minutes=4),
        dropoff_datetime=t0 + timedelta(minutes=14),
    )
    good = dict(base)
    bad_zone = {**base, "DOLocationID": 265}
    bad_time = {**base, "trip_time": 10}
    req_after = {**base, "request_datetime": t0 + timedelta(minutes=10)}
    lyft = {**base, "hvfhs_license_num": "HV0005", "on_scene_datetime": None}
    return [good, bad_zone, bad_time, req_after, lyft]


def test_cleaning_rules_and_null_rates_by_license(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(_rows()).to_parquet(raw / "fhvhv_tripdata_2025-01.parquet")
    con = connect()
    eda.build_base(con, raw)
    eda.build_clean(con)
    dq = eda.data_quality(con)
    assert dq["raw_rows"] == 5 and dq["clean_rows"] == 3  # good, lyft, request-after-pickup
    assert sum(w["dropped_here"] for w in dq["sequential_waterfall"]) == dq["dropped_total"] == 2
    # request-after-pickup is kept for demand/ETA but its wait fields are nulled and flagged
    assert dq["wait_fields_nulled"]["wait_invalid"] == 1
    bad = con.execute("SELECT wait_min, pickup_delay_min FROM clean WHERE wait_invalid").fetchone()
    assert bad == (None, None)
    lic = {r["license"]: r for r in eda.license_breakdown(con)}
    assert lic["HV0005"]["on_scene_null_rate"] == 1.0
    assert lic["HV0003"]["on_scene_null_rate"] == 0.0


def test_run_writes_both_reports(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(_rows()).to_parquet(raw / "fhvhv_tripdata_2025-01.parquet")
    zones = pd.DataFrame(
        {
            "location_id": [4, 5],
            "borough": ["Manhattan"] * 2,
            "zone": ["a", "b"],
            "lat": [40.7, 40.71],
            "x_ft": [0.0, 1000.0],
            "y_ft": [0.0, 0.0],
        }
    )
    zpath = tmp_path / "zones.parquet"
    zones.to_parquet(zpath)
    monkeypatch.setattr(eda.C, "ZONES_PARQUET", zpath)
    out = tmp_path / "reports"
    eda.run(raw, out)
    assert json.loads((out / "data_quality.json").read_text())["raw_rows"] == 5
    assert "geofence_options" in json.loads((out / "eda.json").read_text())
