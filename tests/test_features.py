from datetime import datetime, timedelta

import pandas as pd
import pytest

from fleetopt import constants as C
from fleetopt import features, zones
from fleetopt.db import connect

ZONES = [4, 5, 6]


@pytest.fixture(autouse=True)
def small_geofence(monkeypatch):
    monkeypatch.setattr(zones, "geofence_zone_ids", lambda *a, **k: ZONES)


def _trip(ts, pu, do, servable=True):
    return dict(request_datetime=ts, pu=pu, do_=do, servable=servable)


def _setup(trips, grid=("2025-01-01", "2025-01-20"), splits=None, horizon=1):
    con = connect()
    df = pd.DataFrame(trips, columns=["request_datetime", "pu", "do_", "servable"])
    con.register("t", df)
    con.execute("CREATE TABLE trips AS SELECT * FROM t")
    splits = splits or {"train": ("2025-01-08", "2025-01-15"), "val": ("2025-01-15", "2025-01-20")}
    features.build_demand_features(con, splits, grid, horizon)
    return con


def _frame(con):
    return con.execute("SELECT * FROM demand_features ORDER BY zone, slot_ts").df()


def _trips():
    base = datetime(2025, 1, 10, 12, 0)
    rows = [_trip(base + timedelta(minutes=5 * i), 4, 5) for i in range(20)]
    rows += [_trip(base, 4, 99, servable=False)]
    rows += [_trip(datetime(2025, 1, 3, 9, 1), 5, 6), _trip(datetime(2025, 1, 12, 8, 59), 6, 4)]
    return rows


def test_grid_complete_with_zero_rows():
    df = _frame(_setup(_trips()))
    slots = 19 * C.SLOTS_PER_DAY
    assert len(df) == len(ZONES) * slots
    assert not df.duplicated(["zone", "slot_ts"]).any()
    assert (df["count"] == 0).any()


def test_servable_filter_excludes_out_of_geofence():
    df = _frame(_setup(_trips()))
    assert df["count"].sum() == 22  # 20 + 2 servable singles; the pu=4,do=99 trip is excluded


def test_causality_perturbing_future_leaves_features_unchanged():
    base = _frame(_setup(_trips()))
    t = pd.Timestamp("2025-01-12 12:00")
    future = _trips() + [_trip(t + timedelta(minutes=m), 4, 5) for m in (0, 3, 20, 60, 600)]
    pert = _frame(_setup(future))
    feats = [c for c in base.columns if c.startswith(("lag_", "roll_mean_"))]
    a = base[(base.zone == 4) & (base.slot_ts <= t)][feats].reset_index(drop=True)
    b = pert[(pert.zone == 4) & (pert.slot_ts <= t)][feats].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    # sanity: the count at/after t did change
    assert (base["count"] != pert["count"]).any()


def test_weekly_lag_null_first_week_and_train_excludes_it():
    df = _frame(_setup(_trips()))
    first_week = df[df.slot_ts < "2025-01-08"]
    assert first_week["lag_672"].isna().all()
    assert (df[df.slot_ts >= "2025-01-08"]["lag_672"].notna()).all()
    assert (first_week["split"].isna()).all()
    assert df[df.split == "train"].slot_ts.min() == pd.Timestamp("2025-01-08")


def test_lag_values_align_to_wall_clock():
    df = _frame(_setup(_trips())).set_index(["zone", "slot_ts"])
    t = pd.Timestamp("2025-01-10 12:30")
    assert df.loc[(4, t), "lag_1"] == df.loc[(4, t - pd.Timedelta(minutes=15)), "count"]
    assert df.loc[(4, t), "lag_96"] == df.loc[(4, t - pd.Timedelta(days=1)), "count"]
    # rolling mean excludes the current slot
    prev = [df.loc[(4, t - pd.Timedelta(minutes=15 * k)), "count"] for k in range(1, 5)]
    assert df.loc[(4, t), "roll_mean_4"] == pytest.approx(sum(prev) / 4)


def test_horizon_2_drops_lag_1_and_shifts_rolling():
    df = _frame(_setup(_trips(), horizon=2)).set_index(["zone", "slot_ts"])
    assert "lag_1" not in df.columns and "lag_4" in df.columns
    t = pd.Timestamp("2025-01-10 12:30")
    prev = [df.loc[(4, t - pd.Timedelta(minutes=15 * k)), "count"] for k in range(2, 6)]
    assert df.loc[(4, t), "roll_mean_4"] == pytest.approx(sum(prev) / 4)
    assert (df["horizon"] == 2).all()


def test_dst_gap_rows_null_and_excluded():
    # US spring-forward 2025-03-09: 02:00-02:59 local does not exist.
    grid = ("2025-03-08", "2025-03-11")
    splits = {"train": ("2025-03-08", "2025-03-11")}
    df = _frame(_setup([_trip(datetime(2025, 3, 9, 12), 4, 5)], grid, splits))
    gap = df[df.is_dst_gap]
    assert len(gap) == 4 * len(ZONES)
    assert gap["count"].isna().all() and gap["split"].isna().all()
    assert (df[~df.is_dst_gap]["count"].notna()).all()
    # windows skip the NULL slots: a slot with only some gap slots in its window still gets a mean
    after = df[(df.zone == 4) & (df.slot_ts == "2025-03-09 03:15")]
    assert after["roll_mean_4"].notna().all()


def test_dst_gap_slots_none_outside_spring_forward():
    assert features.dst_gap_slots("2025-01-01", "2025-02-01") == []
    assert len(features.dst_gap_slots("2025-03-09", "2025-03-10")) == 4


def test_zone_mean_rate_uses_train_only():
    trips = _trips()
    a = _frame(_setup(trips))
    # add val-period demand; train-only zone rate must not move
    b = _frame(_setup(trips + [_trip(datetime(2025, 1, 16, 10), 4, 5) for _ in range(50)]))
    ra = a.groupby("zone").zone_mean_rate.first()
    rb = b.groupby("zone").zone_mean_rate.first()
    pd.testing.assert_series_equal(ra, rb)


def test_horizon_must_be_positive():
    with pytest.raises(ValueError):
        _setup(_trips(), horizon=0)
