"""Train the ETA and demand models, the travel matrix, and write reports/evaluation.json.

Selection (hyper-parameters, early stopping) uses the validation window only; the test window is
scored exactly once per model after selection. No refit on train+val (documented limitation).
"""

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_poisson_deviance

from zone_ranger import artifacts as A
from zone_ranger import constants as C
from zone_ranger import zones
from zone_ranger.db import connect
from zone_ranger.travel import TravelMatrix, band_of_hour

ETA_SAMPLE_ROWS = 2_000_000
ETA_VAL_ES_ROWS = 1_000_000  # val rows used for early stopping / selection; metrics use all val
BASE_ETA_FEATURES = [
    "pu",
    "do_",
    "hour",
    "slot_of_day",
    "dow",
    "is_weekend",
    "us_holiday",
    "centroid_km",
    "route_median",
    "pu_median",
    "do_median",
    "route_band_median",
]
LIMITATIONS = [
    "No refit on train+val: reported test numbers come from a model trained on Jan 8 - Feb 28.",
    "Demand counts only servable (intra-geofence) requests; out-of-geofence demand is invisible.",
    "Waits are Uber-only and completed-trips-only (Lyft on_scene starts 2025-03-06).",
    "Travel bands ignore weekday/weekend; fallback cells use haversine x citywide median speed.",
    "ETA route medians are fit on the same sampled train rows the model trains on (min count 20).",
    "Naive local timestamps: the 2025-03-09 DST gap is NULL and excluded; no fall-back in window.",
    "Test window ends 30 min before May 1: later requests are right-censored in the April file.",
    "ETA trip_miles ablation uses a post-trip quantity and is an upper bound, not deployable.",
]
_test_calls: Counter = Counter()


def score_test(name: str, fn):
    """Run `fn` on the test window; refuses a second look within one run."""
    _test_calls[name] += 1
    if _test_calls[name] > 1:
        raise RuntimeError(f"test window already scored for {name}")
    return fn()


def reg_metrics(y, p) -> dict:
    err = np.abs(np.asarray(y, float) - np.asarray(p, float))
    return {
        "n": int(len(err)),
        "mae": float(err.mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "p90_abs_error": float(np.quantile(err, 0.9)),
    }


def demand_metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = reg_metrics(y, p)
    m["wmape"] = float(np.abs(y - p).sum() / max(y.sum(), 1e-9))
    m["poisson_deviance"] = float(mean_poisson_deviance(y, np.clip(p, 1e-6, None)))
    m["pred_to_actual_total"] = float(p.sum() / max(y.sum(), 1e-9))
    return m


def sliced(df: pd.DataFrame, key: pd.Series, metric_fn) -> dict:
    out = {}
    for k, g in df.groupby(key, observed=True):
        out[str(k)] = metric_fn(g.y, g.p)
    return out


def _band(hour: pd.Series) -> pd.Series:
    return hour.map(band_of_hour)


# ------------------------------------------------------------------ ETA


def _eta_query(window: tuple[str, str], sample: int | None) -> str:
    cols = (
        "pu, do_, hour, slot_of_day, dow, CAST(is_weekend AS INT) AS is_weekend, "
        "CAST(us_holiday AS INT) AS us_holiday, trip_miles, duration_min"
    )
    q = (
        f"SELECT {cols} FROM trips WHERE servable AND request_datetime >= TIMESTAMP '{window[0]}' "
        f"AND request_datetime < TIMESTAMP '{window[1]}'"
    )
    if sample:
        q = f"SELECT * FROM ({q}) USING SAMPLE reservoir({sample} ROWS) REPEATABLE ({C.SEED})"
    return q


class _Centroids:
    def __init__(self, zone_ids: list[int]):
        cen = zones.centroids(zone_ids)
        self.lat = dict(zip(cen.zone, cen.lat, strict=True))
        self.lon = dict(zip(cen.zone, cen.lon, strict=True))

    def km(self, pu: pd.Series, do: pd.Series) -> np.ndarray:
        return zones.haversine_km(
            pu.map(self.lat).to_numpy(float),
            pu.map(self.lon).to_numpy(float),
            do.map(self.lat).to_numpy(float),
            do.map(self.lon).to_numpy(float),
        )


def eta_frame(
    df: pd.DataFrame,
    enc: A.EtaFeatureEncoder,
    cen: _Centroids,
    zone_ids: list[int],
    with_miles: bool,
) -> pd.DataFrame:
    x = df[["pu", "do_", "hour", "slot_of_day", "dow", "is_weekend", "us_holiday"]].copy()
    x["centroid_km"] = cen.km(df.pu, df.do_)
    x = pd.concat([x, enc.transform(df)], axis=1)
    if with_miles:
        x["trip_miles"] = df.trip_miles.to_numpy()
    for c in ("pu", "do_"):
        x[c] = pd.Categorical(x[c], categories=zone_ids)
    return x


def _fit_lgb(params, grid, xtr, ytr, xes, yes, rounds, fast) -> tuple[lgb.Booster, dict, float]:
    """Small grid with early stopping on the selection set; returns best (model, params, score)."""
    best = (None, None, np.inf)
    for combo in itertools.product(*grid.values()):
        p = {**params, **dict(zip(grid, combo, strict=True)), "seed": C.SEED, "verbosity": -1}
        dtr = lgb.Dataset(xtr, ytr, free_raw_data=False)
        des = lgb.Dataset(xes, yes, reference=dtr)
        model = lgb.train(
            p,
            dtr,
            num_boost_round=rounds,
            valid_sets=[des],
            callbacks=[lgb.early_stopping(20 if fast else 30, verbose=False)],
        )
        score = model.best_score["valid_0"][p["metric"]]
        if score < best[2]:
            best = (model, p, float(score))
    return best


def train_eta(con, splits, zone_ids, cen, fast: bool) -> tuple[dict, A.EtaBundle]:
    sample = None if fast else ETA_SAMPLE_ROWS
    tr = con.execute(_eta_query(splits["train"], sample)).df()
    enc = A.EtaFeatureEncoder().fit(tr)
    va = con.execute(_eta_query(splits["val"], None)).df()
    te = con.execute(_eta_query(splits["test"], None)).df()
    es = va.sample(min(len(va), ETA_VAL_ES_ROWS), random_state=C.SEED)

    base = {"objective": "l1", "metric": "l1", "num_threads": 0}
    grid = {"num_leaves": [63, 255], "learning_rate": [0.1], "min_data_in_leaf": [100, 500]}
    if fast:
        grid = {"num_leaves": [15, 31], "learning_rate": [0.1], "min_data_in_leaf": [20]}
    rounds = 40 if fast else 600
    cuts = tr.duration_min.quantile([1 / 3, 2 / 3]).to_numpy()

    def tercile(d: pd.DataFrame) -> pd.Series:
        return pd.Series(np.digitize(d.duration_min, cuts), index=d.index).map(
            {0: "short", 1: "medium", 2: "long"}
        )

    def evaluate(d, p) -> dict:
        r = d.assign(y=d.duration_min, p=p)
        out = reg_metrics(r.y, r.p)
        out["by_time_band"] = sliced(r, _band(r.hour), reg_metrics)
        out["by_trip_length_tercile"] = sliced(r, tercile(r), reg_metrics)
        return out

    results, bundle_model = {}, None
    for name, with_miles in (("main", False), ("ablation_trip_miles", True)):
        xtr, xes = (eta_frame(d, enc, cen, zone_ids, with_miles) for d in (tr, es))
        model, params, score = _fit_lgb(
            base, grid, xtr, tr.duration_min, xes, es.duration_min, rounds, fast
        )
        pv = model.predict(eta_frame(va, enc, cen, zone_ids, with_miles))
        pt = score_test(
            f"eta_{name}",
            lambda m=model, w=with_miles: m.predict(eta_frame(te, enc, cen, zone_ids, w)),
        )
        results[name] = {
            "params": params,
            "best_iteration": model.best_iteration,
            "val": evaluate(va, pv),
            "test": evaluate(te, pt),
            "features": list(xtr.columns),
        }
        if name == "main":
            bundle_model, feats = model, list(xtr.columns)

    base_val = {"route_median": enc.route_median(va.pu, va.do_), "global_median": enc.global_median}
    base_test = {
        "route_median": enc.route_median(te.pu, te.do_),
        "global_median": enc.global_median,
    }
    results["baselines"] = {
        k: {
            "val": evaluate(va, np.broadcast_to(base_val[k], len(va))),
            "test": evaluate(te, np.broadcast_to(base_test[k], len(te))),
        }
        for k in base_val
    }
    results["rows"] = {"train": len(tr), "val": len(va), "test": len(te)}
    bundle = A.EtaBundle(
        model=bundle_model,
        metrics={"val": results["main"]["val"], "test": results["main"]["test"]},
        train_window=splits["train"],
        data_fingerprint=A.fingerprint(results["rows"], splits),
        encoder=enc,
        feature_names=feats,
        uses_trip_miles=False,
    )
    return results, bundle


# ------------------------------------------------------------------ demand


def train_demand(con, splits, zone_ids, fast: bool) -> tuple[dict, A.DemandBundle]:
    df = con.execute("SELECT * FROM demand_features WHERE split IS NOT NULL").df()
    horizon = int(df.horizon.iloc[0])
    for c in ("is_weekend", "us_holiday"):
        df[c] = df[c].astype(int)
    for c in df.columns:
        if c.startswith("lag_"):
            df[c] = df[c].astype("float64")  # nullable Int64 from DuckDB breaks float fillna
    df["hour"] = df.slot_of_day // (60 // C.SLOT_MINUTES)
    feats = (
        [f"lag_{k}" for k in C.DEMAND_LAGS if k >= horizon]
        + [f"roll_mean_{n}" for n in C.DEMAND_ROLLING]
        + ["slot_of_day", "dow", "is_weekend", "us_holiday", "zone_mean_rate", "zone"]
    )
    df["zone"] = pd.Categorical(df.zone, categories=zone_ids)
    parts = {s: df[df.split == s] for s in ("train", "val", "test")}
    tr, va, te = parts["train"], parts["val"], parts["test"]

    base = {"objective": "poisson", "metric": "poisson", "num_threads": 0}
    grid = {"num_leaves": [31, 127], "learning_rate": [0.05, 0.1], "min_data_in_leaf": [50]}
    if fast:
        grid = {"num_leaves": [7, 15], "learning_rate": [0.1], "min_data_in_leaf": [20]}
    model, params, _ = _fit_lgb(
        base, grid, tr[feats], tr["count"], va[feats], va["count"], 60 if fast else 1000, fast
    )

    tiers = pd.qcut(
        tr.groupby("zone", observed=True).zone_mean_rate.first().rank(method="first"),
        3,
        labels=["low", "mid", "high"],
    )

    def evaluate(d: pd.DataFrame, p) -> dict:
        r = d.assign(y=d["count"], p=p)
        out = demand_metrics(r.y, r.p)
        out["by_time_band"] = sliced(r, _band(r.slot_of_day // 4), demand_metrics)
        out["by_zone_volume_tier"] = sliced(r, r.zone.map(tiers), demand_metrics)
        return out

    def last_week(d):  # same slot last week; falls back to yesterday when the weekly lag is NaN
        lw = d["lag_672"] if "lag_672" in d else pd.Series(np.nan, index=d.index)
        return lw.fillna(d["lag_96"]).fillna(d.zone_mean_rate).to_numpy()

    def yesterday(d):
        return d["lag_96"].fillna(d.zone_mean_rate).to_numpy()

    pv = model.predict(va[feats])
    pt = score_test("demand", lambda: model.predict(te[feats]))
    results = {
        "horizon": horizon,
        "params": params,
        "best_iteration": model.best_iteration,
        "features": feats,
        "val": evaluate(va, pv),
        "test": evaluate(te, pt),
        "baselines": {
            "same_slot_last_week": {
                "val": evaluate(va, last_week(va)),
                "test": evaluate(te, last_week(te)),
            },
            "same_slot_yesterday": {
                "val": evaluate(va, yesterday(va)),
                "test": evaluate(te, yesterday(te)),
            },
        },
        "rows": {k: len(v) for k, v in parts.items()},
    }
    bundle = A.DemandBundle(
        model=model,
        metrics={"val": results["val"], "test": results["test"]},
        train_window=splits["train"],
        data_fingerprint=A.fingerprint(results["rows"], splits),
        feature_names=feats,
        horizon=horizon,
        zone_ids=zone_ids,
    )
    return results, bundle


# ------------------------------------------------------------------ driver


def _stage(artifacts_dir: Path, name: str, splits: dict, force: bool, build, save, load):
    """Checkpointed stage: reuse artifacts from a previous run unless `force` or the splits changed.

    `build()` returns (results, artifact); `save(artifact)` / `load()` persist it. The results
    JSON is written last, so a run killed mid-stage leaves no checkpoint behind.
    """
    meta = artifacts_dir / f"{name}_results.json"
    wanted = {k: list(v) for k, v in splits.items()}
    if not force and meta.exists():
        saved = json.loads(meta.read_text())
        if saved["splits"] == wanted:
            try:
                art = load()
            except FileNotFoundError:
                art = None
            if art is not None:
                print(f"[train] {name}: reusing checkpoint (use --force to retrain)", flush=True)
                return saved["results"], art
    results, art = build()
    save(art)
    meta.write_text(json.dumps({"splits": wanted, "results": results}))
    return results, art


def train_all(
    con: duckdb.DuckDBPyConnection,
    splits: dict[str, tuple[str, str]],
    zone_ids: list[int],
    artifacts_dir: Path,
    report_path: Path,
    fast: bool = False,
    force: bool = False,
) -> dict:
    _test_calls.clear()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cen = _Centroids(zone_ids)

    def build_travel():
        tm_trips = con.execute(
            f"""SELECT pu, do_, hour, duration_min, trip_miles FROM trips WHERE servable
                AND request_datetime >= TIMESTAMP '{splits["train"][0]}'
                AND request_datetime < TIMESTAMP '{splits["train"][1]}'"""
        ).df()
        tm = TravelMatrix.fit(tm_trips, zone_ids, min_trips=5 if fast else C.TRAVEL_MIN_TRIPS)
        return {}, tm

    path = artifacts_dir / "travel.joblib"
    _, travel = _stage(
        artifacts_dir, "travel", splits, force, build_travel,
        lambda a: joblib.dump(a, path), lambda: joblib.load(path),
    )  # fmt: skip
    path = artifacts_dir / "eta_bundle.joblib"
    eta, _ = _stage(
        artifacts_dir, "eta", splits, force, lambda: train_eta(con, splits, zone_ids, cen, fast),
        lambda b: A.save_bundle(b, path), lambda: A.load_eta_bundle(path),
    )  # fmt: skip
    path = artifacts_dir / "demand_bundle.joblib"
    demand, _ = _stage(
        artifacts_dir, "demand", splits, force, lambda: train_demand(con, splits, zone_ids, fast),
        lambda b: A.save_bundle(b, path), lambda: A.load_demand_bundle(path),
    )  # fmt: skip

    report = {
        "splits": {k: list(v) for k, v in splits.items()},
        "geofence_zones": len(zone_ids),
        "eta": eta,
        "demand": demand,
        "travel_fallback_coverage": travel.coverage,
        "travel_median_speed_mph": travel.median_speed_mph,
        "limitations": LIMITATIONS,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return report


def run(dev: bool = False, force: bool = False) -> dict:
    con = connect(C.DEV_DUCKDB_PATH if dev else C.DUCKDB_PATH)
    splits = C.DEV_SPLITS if dev else C.SPLITS
    art = C.ARTIFACTS_DIR / "dev" if dev else C.ARTIFACTS_DIR
    rep = C.REPORTS_DIR / ("evaluation_dev.json" if dev else "evaluation.json")
    return train_all(con, splits, zones.geofence_zone_ids(), art, rep, fast=dev, force=force)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", action="store_true", help="small grids; uses the dev database")
    ap.add_argument("--force", action="store_true", help="retrain every stage, ignore checkpoints")
    args = ap.parse_args()
    report = run(dev=args.dev, force=args.force)
    d, e = report["demand"], report["eta"]
    print(
        f"demand test WMAPE {d['test']['wmape']:.4f} vs last-week "
        f"{d['baselines']['same_slot_last_week']['test']['wmape']:.4f}\n"
        f"ETA test MAE {e['main']['test']['mae']:.3f} vs route median "
        f"{e['baselines']['route_median']['test']['mae']:.3f}"
    )


if __name__ == "__main__":
    main()
