"""Serializable model artifacts: the ETA feature encoder and model/demand bundles.

Everything fitted here (medians, zone rates) uses the train window only. Bundles carry enough
metadata (version, metrics, windows, data fingerprint) to know what produced a prediction.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fleetopt import constants as C
from fleetopt.travel import band_of_hour

MIN_COUNT = 20  # smallest group whose median is trusted


def _version() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


class EtaFeatureEncoder:
    """Train-only median durations with a fallback chain.

    Columns produced (all request-time knowable): route_median, pu_median, do_median,
    route_band_median. Fallbacks: route-band -> route -> pickup zone -> global median;
    route -> pickup zone -> global; pu/do -> global.
    """

    def __init__(self, min_count: int = MIN_COUNT):
        self.min_count = min_count
        self.fitted = False

    def _table(self, df: pd.DataFrame, keys: list[str]) -> pd.Series:
        g = df.groupby(keys).duration_min.agg(["median", "size"])
        return g[g["size"] >= self.min_count]["median"]

    def fit(self, df: pd.DataFrame) -> "EtaFeatureEncoder":
        """`df` needs pu, do_, hour, duration_min (train rows only)."""
        df = df.assign(band=df.hour.map(band_of_hour))
        self.global_median = float(df.duration_min.median())
        self.pu = self._table(df, ["pu"]).to_dict()
        self.do = self._table(df, ["do_"]).to_dict()
        self.route = self._table(df, ["pu", "do_"]).to_dict()
        self.route_band = self._table(df, ["pu", "do_", "band"]).to_dict()
        self.fitted = True
        return self

    def route_median(self, pu: pd.Series, do: pd.Series) -> np.ndarray:
        """Route median with fallback; this is also the 'route median' baseline."""
        return np.array(
            [
                self.route.get((p, d), self.pu.get(p, self.global_median))
                for p, d in zip(pu, do, strict=True)
            ]
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("encoder is not fitted")
        route = self.route_median(df.pu, df.do_)
        band = df.hour.map(band_of_hour)
        rb = [
            self.route_band.get((p, d, b), r)
            for p, d, b, r in zip(df.pu, df.do_, band, route, strict=True)
        ]
        return pd.DataFrame(
            {
                "route_median": route,
                "pu_median": df.pu.map(self.pu).fillna(self.global_median).to_numpy(),
                "do_median": df.do_.map(self.do).fillna(self.global_median).to_numpy(),
                "route_band_median": np.array(rb),
            },
            index=df.index,
        )

    def save(self, path: Path) -> None:
        joblib.dump(self, path)


@dataclass
class _Bundle:
    model: Any
    metrics: dict
    train_window: tuple[str, str]
    data_fingerprint: dict
    version: str = field(default_factory=_version)
    extra: dict = field(default_factory=dict)


@dataclass
class EtaBundle(_Bundle):
    encoder: EtaFeatureEncoder | None = None
    feature_names: list[str] = field(default_factory=list)
    uses_trip_miles: bool = False


@dataclass
class DemandBundle(_Bundle):
    feature_names: list[str] = field(default_factory=list)
    horizon: int = C.DEMAND_HORIZON
    zone_ids: list[int] = field(default_factory=list)


def fingerprint(row_counts: dict[str, int], splits: dict[str, tuple[str, str]]) -> dict:
    """Git-independent identity of the data a bundle was trained on."""
    return {"row_counts": row_counts, "splits": {k: list(v) for k, v in splits.items()}}


def save_bundle(bundle: _Bundle, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def _load(path: Path, cls: type) -> Any:
    obj = joblib.load(path)
    if not isinstance(obj, cls):
        raise TypeError(f"{path} holds a {type(obj).__name__}, expected {cls.__name__}")
    return obj


def load_eta_bundle(path: Path) -> EtaBundle:
    return _load(path, EtaBundle)


def load_demand_bundle(path: Path) -> DemandBundle:
    return _load(path, DemandBundle)
