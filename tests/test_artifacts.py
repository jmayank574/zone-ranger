import numpy as np
import pandas as pd
import pytest

from zone_ranger import artifacts as A


def _train():
    rows = []
    # route 1->2: 30 trips, 10 min (hour 8 = am_peak) / 20 min (hour 3 = overnight)
    rows += [dict(pu=1, do_=2, hour=8, duration_min=20.0)] * 25
    rows += [dict(pu=1, do_=2, hour=3, duration_min=10.0)] * 25
    # pickup zone 1 has other destinations; route 1->3 is sparse (below min count)
    rows += [dict(pu=1, do_=3, hour=8, duration_min=40.0)] * 5
    rows += [dict(pu=4, do_=5, hour=12, duration_min=7.0)] * 30
    return pd.DataFrame(rows)


def test_fallback_chain_order():
    enc = A.EtaFeatureEncoder(min_count=20).fit(_train())
    q = pd.DataFrame(
        {
            "pu": [1, 1, 9, 4],
            "do_": [2, 3, 9, 5],
            "hour": [8, 8, 8, 12],
        }
    )
    out = enc.transform(q)
    assert out.route_band_median[0] == 20.0  # route-band exists
    assert out.route_median[1] == enc.pu[1]  # sparse route -> pickup-zone median
    assert out.route_band_median[1] == enc.pu[1]
    assert out.route_median[2] == enc.global_median  # unseen zone -> global
    assert out.pu_median[2] == enc.global_median
    assert out.route_median[3] == 7.0


def test_encoder_fits_on_train_only():
    train = _train()
    enc = A.EtaFeatureEncoder().fit(train)
    q = pd.DataFrame({"pu": [1], "do_": [2], "hour": [8]})
    before = enc.transform(q).route_median[0]
    # val rows with very different durations must not move the fitted medians
    val = pd.DataFrame([dict(pu=1, do_=2, hour=8, duration_min=999.0)] * 500)
    _ = enc.transform(val)  # transform never updates state
    assert enc.transform(q).route_median[0] == before
    refit = A.EtaFeatureEncoder().fit(pd.concat([train, val]))
    assert refit.transform(q).route_median[0] != before


def test_unfitted_encoder_raises():
    with pytest.raises(RuntimeError):
        A.EtaFeatureEncoder().transform(pd.DataFrame({"pu": [1], "do_": [2], "hour": [1]}))


def test_bundle_round_trip_and_type_check(tmp_path):
    enc = A.EtaFeatureEncoder().fit(_train())
    fp = A.fingerprint({"train": 10}, {"train": ("2025-01-08", "2025-03-01")})
    eta = A.EtaBundle(
        model={"stub": 1},
        metrics={"mae": 1.0},
        train_window=("2025-01-08", "2025-03-01"),
        data_fingerprint=fp,
        encoder=enc,
        feature_names=["pu"],
    )
    p = A.save_bundle(eta, tmp_path / "eta.joblib")
    back = A.load_eta_bundle(p)
    assert back.version == eta.version and back.metrics == {"mae": 1.0}
    np.testing.assert_array_equal(
        back.encoder.transform(_train()).to_numpy(), enc.transform(_train()).to_numpy()
    )
    with pytest.raises(TypeError):
        A.load_demand_bundle(p)
