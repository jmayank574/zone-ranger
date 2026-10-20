import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from zone_ranger import artifacts, features, train, zones
from zone_ranger.db import connect

ZONES = [4, 5, 6]
CEN = pd.DataFrame({"zone": ZONES, "lat": [40.70, 40.75, 40.80], "lon": [-74.0, -73.98, -73.96]})
SPLITS = {
    "train": ("2025-01-08", "2025-01-15"),
    "val": ("2025-01-15", "2025-01-18"),
    "test": ("2025-01-18", "2025-01-22"),
}
GRID = ("2025-01-01", "2025-01-22")


@pytest.fixture
def synthetic_con(monkeypatch):
    monkeypatch.setattr(zones, "geofence_zone_ids", lambda *a, **k: ZONES)
    monkeypatch.setattr(zones, "centroids", lambda ids: CEN)
    rng = np.random.default_rng(0)
    rows, t0 = [], datetime(2025, 1, 1)
    for _ in range(6000):
        ts = t0 + timedelta(minutes=int(rng.integers(0, 21 * 24 * 60)))
        pu, do = int(rng.choice(ZONES)), int(rng.choice(ZONES))
        miles = float(rng.uniform(0.5, 6))
        rows.append(
            dict(
                request_datetime=ts,
                pu=pu,
                do_=do,
                servable=True,
                hour=ts.hour,
                slot_of_day=(ts.hour * 60 + ts.minute) // 15,
                dow=ts.weekday(),
                is_weekend=ts.weekday() >= 5,
                us_holiday=False,
                trip_miles=miles,
                duration_min=3 + 4 * miles + float(rng.normal(0, 1)),
            )
        )
    con = connect()
    con.register("t", pd.DataFrame(rows))
    con.execute("CREATE TABLE trips AS SELECT * FROM t")
    features.build_demand_features(con, SPLITS, GRID)
    return con


def test_end_to_end_smoke(synthetic_con, tmp_path):
    rep = train.train_all(synthetic_con, SPLITS, ZONES, tmp_path / "art", tmp_path / "e.json", True)
    assert set(rep) >= {"eta", "demand", "travel_fallback_coverage", "limitations", "splits"}
    for k in ("main", "ablation_trip_miles", "baselines", "rows"):
        assert k in rep["eta"]
    assert set(rep["eta"]["baselines"]) == {"route_median", "global_median"}
    assert set(rep["demand"]["baselines"]) == {"same_slot_last_week", "same_slot_yesterday"}
    for k in ("mae", "rmse", "wmape", "poisson_deviance", "pred_to_actual_total"):
        assert k in rep["demand"]["test"]
    assert "by_time_band" in rep["eta"]["main"]["test"]
    assert "by_zone_volume_tier" in rep["demand"]["test"]
    # trip_miles is a leaky feature: the ablation should beat the deployable model
    assert rep["eta"]["ablation_trip_miles"]["test"]["mae"] < rep["eta"]["main"]["test"]["mae"]
    assert json.loads((tmp_path / "e.json").read_text())["geofence_zones"] == 3
    art = tmp_path / "art"
    assert artifacts.load_eta_bundle(art / "eta_bundle.joblib").uses_trip_miles is False
    assert artifacts.load_demand_bundle(art / "demand_bundle.joblib").horizon == 1
    assert (art / "travel.joblib").exists()


def test_test_window_scored_once():
    train._test_calls.clear()
    assert train.score_test("x", lambda: 1) == 1
    with pytest.raises(RuntimeError):
        train.score_test("x", lambda: 2)
    train._test_calls.clear()


def test_metrics_definitions():
    m = train.demand_metrics([0, 2, 4], [1, 2, 3])
    assert m["mae"] == pytest.approx(2 / 3)
    assert m["wmape"] == pytest.approx(2 / 6)
    assert m["pred_to_actual_total"] == pytest.approx(1.0)
    assert train.reg_metrics([1, 2], [1, 4])["rmse"] == pytest.approx(np.sqrt(2))


def test_null_integer_lag_columns_train(synthetic_con, tmp_path):
    """DuckDB returns lag_* as nullable Int64; the NULL weekly lag in the first week must not break
    float fillna in the baselines or LightGBM."""
    # wipe the weekly and daily lags early in the window so NULLs reach train/val rows
    synthetic_con.execute(
        "UPDATE demand_features SET lag_672 = NULL, lag_96 = NULL WHERE slot_ts < '2025-01-16'"
    )
    df = synthetic_con.execute("SELECT * FROM demand_features WHERE split IS NOT NULL").df()
    assert str(df["lag_672"].dtype) == "Int64" and df["lag_672"].isna().any()
    out, _ = train.train_demand(synthetic_con, SPLITS, ZONES, fast=True)
    assert out["val"]["wmape"] >= 0
    assert out["baselines"]["same_slot_last_week"]["val"]["wmape"] >= 0


def test_checkpointing_skips_and_force_retrains(synthetic_con, tmp_path, monkeypatch):
    args = (synthetic_con, SPLITS, ZONES, tmp_path / "art", tmp_path / "e.json", True)
    first = train.train_all(*args)
    calls = []
    for name in ("train_eta", "train_demand"):
        real = getattr(train, name)
        monkeypatch.setattr(
            train, name, lambda *a, _r=real, _n=name, **k: calls.append(_n) or _r(*a, **k)
        )
    second = train.train_all(*args)
    assert calls == []  # every stage reused
    assert second["eta"]["main"]["test"] == first["eta"]["main"]["test"]
    assert second["demand"]["test"] == first["demand"]["test"]
    (tmp_path / "art" / "demand_bundle.joblib").unlink()  # partial checkpoint: only demand reruns
    train.train_all(*args)
    assert calls == ["train_demand"]
    train.train_all(*args, force=True)
    assert calls == ["train_demand", "train_eta", "train_demand"]
