import numpy as np
import pandas as pd
import pytest

from zone_ranger import travel, zones
from zone_ranger.travel import TravelMatrix

ZONES = [4, 5, 6]
CEN = pd.DataFrame({"zone": ZONES, "lat": [40.70, 40.75, 40.80], "lon": [-74.0, -73.98, -73.96]})


@pytest.fixture(autouse=True)
def fake_centroids(monkeypatch):
    monkeypatch.setattr(zones, "centroids", lambda ids: CEN)


def _trips():
    rows = []
    # 4->5: dense in am_peak (hour 8) and overnight (hour 2); midday sparse; pm absent
    rows += [dict(pu=4, do_=5, hour=8, duration_min=15.0, trip_miles=3.0)] * 25
    rows += [dict(pu=4, do_=5, hour=2, duration_min=9.0, trip_miles=3.0)] * 25
    rows += [dict(pu=4, do_=5, hour=12, duration_min=99.0, trip_miles=3.0)] * 3
    # pickup zone 5 has only a sparse trip to 6, but enough in total for the pu median
    rows += [dict(pu=5, do_=4, hour=8, duration_min=12.0, trip_miles=2.0)] * 30
    return pd.DataFrame(rows)


def test_band_assignment():
    assert travel.band_of_hour(0) == "overnight" and travel.band_of_hour(5) == "overnight"
    assert travel.band_of_hour(6) == "am_peak" and travel.band_of_hour(9) == "am_peak"
    assert travel.band_of_hour(10) == "midday" and travel.band_of_hour(15) == "midday"
    assert travel.band_of_hour(16) == "pm_evening" and travel.band_of_hour(23) == "pm_evening"
    with pytest.raises(ValueError):
        travel.band_of_hour(24)


def test_fallback_levels_hit_as_expected():
    tm = TravelMatrix.fit(_trips(), ZONES, min_trips=20)
    assert tm.level_of(4, 5, "am_peak") == "pair_band"
    assert tm.minutes(4, 5, "am_peak") == 15.0
    assert tm.minutes(4, 5, "overnight") == 9.0
    # midday pair has only 3 trips -> all-day pair median over all 53 trips
    assert tm.level_of(4, 5, "midday") == "pair_all_day"
    assert tm.minutes(4, 5, "midday") == pytest.approx(15.0)
    # pm band has no 4->5 trips -> pair all-day
    assert tm.level_of(4, 5, "pm_evening") == "pair_all_day"
    # 5->6 never observed but pickup zone 5 is dense
    assert tm.level_of(5, 6, "am_peak") == "pickup_zone"
    assert tm.minutes(5, 6, "am_peak") == 12.0
    # zone 6 never a pickup -> haversine
    assert tm.level_of(6, 4, "am_peak") == "haversine"
    assert tm.minutes(6, 4, "am_peak") > 0


def test_matrix_shape_coverage_and_asymmetry_allowed():
    tm = TravelMatrix.fit(_trips(), ZONES, min_trips=20)
    for band in travel.band_names():
        m = tm.matrix(band)
        assert m.shape == (3, 3) and np.isfinite(m).all() and (m > 0).all()
        assert tm.miles_matrix(band).shape == (3, 3)
        assert sum(tm.coverage[band].values()) == pytest.approx(1.0)
    assert tm.minutes(4, 5, "am_peak") != tm.minutes(5, 4, "am_peak")  # asymmetric
