"""Zone-to-zone travel time and distance by time-of-day band, from the train window only.

Bands are weekday/weekend-agnostic in v1. Each cell falls back through
pair-band -> pair all-day -> pickup-zone median -> haversine x citywide median speed.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fleetopt import constants as C
from fleetopt import zones

LEVELS = ("pair_band", "pair_all_day", "pickup_zone", "haversine")
KM_PER_MILE = 1.609344
MIN_FALLBACK_MILES = 0.3  # floor so same-zone (zero centroid distance) cells are not 0 minutes


def band_of_hour(hour: int) -> str:
    for name, lo, hi in C.TIME_BANDS:
        if lo <= hour < hi:
            return name
    raise ValueError(f"hour out of range: {hour}")


def band_names() -> list[str]:
    return [b[0] for b in C.TIME_BANDS]


def _med(df: pd.DataFrame, keys: list[str], min_trips: int) -> pd.DataFrame:
    g = df.groupby(keys).agg(
        minutes=("duration_min", "median"), miles=("trip_miles", "median"), n=("pu", "size")
    )
    return g[g.n >= min_trips].drop(columns="n")


@dataclass
class TravelMatrix:
    zone_ids: list[int]
    bands: dict[str, dict[str, np.ndarray]]  # band -> {"minutes", "miles", "level"} (n x n)
    median_speed_mph: float
    min_trips: int
    coverage: dict = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        trips: pd.DataFrame,
        zone_ids: list[int] | None = None,
        min_trips: int = C.TRAVEL_MIN_TRIPS,
    ) -> "TravelMatrix":
        """`trips` needs pu, do_, hour, duration_min, trip_miles (intra-geofence, train only)."""
        zone_ids = zone_ids if zone_ids is not None else zones.geofence_zone_ids()
        n = len(zone_ids)
        trips = trips[trips.pu.isin(zone_ids) & trips.do_.isin(zone_ids)].copy()
        trips["band"] = trips.hour.map(band_of_hour)
        speed = float((trips.trip_miles / (trips.duration_min / 60.0)).median())
        cen = zones.centroids(zone_ids)
        lat, lon = cen.lat.to_numpy(), cen.lon.to_numpy()
        hav_mi = zones.haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
        hav_mi = np.maximum(hav_mi / KM_PER_MILE, MIN_FALLBACK_MILES)
        hav_min = hav_mi / speed * 60.0

        pos = {z: i for i, z in enumerate(zone_ids)}
        pair_all = _med(trips, ["pu", "do_"], min_trips)
        pu_all = _med(trips, ["pu"], min_trips)
        out, cover = {}, {}
        for band in band_names():
            pair_band = _med(trips[trips.band == band], ["pu", "do_"], min_trips)
            mins, miles = hav_min.copy(), hav_mi.copy()
            level = np.full((n, n), 3, dtype=np.int8)
            # apply from the coarsest fallback to the finest so finer levels overwrite
            for lvl, table, keyed_by_pair in (
                (2, pu_all, False),
                (1, pair_all, True),
                (0, pair_band, True),
            ):
                for key, row in table.iterrows():
                    if keyed_by_pair:
                        i, j = pos[key[0]], pos[key[1]]
                        sel = (np.array([i]), np.array([j]))
                    else:
                        i = pos[key]
                        sel = (np.full(n, i), np.arange(n))
                        # only fill cells still at the haversine level
                        keep = level[sel] == 3
                        sel = (sel[0][keep], sel[1][keep])
                    mins[sel], miles[sel], level[sel] = row.minutes, row.miles, lvl
            out[band] = {"minutes": mins, "miles": miles, "level": level}
            cover[band] = {name: float((level == k).mean()) for k, name in enumerate(LEVELS)}
        return cls(zone_ids, out, speed, min_trips, cover)

    def _pos(self, zone: int) -> int:
        return self.zone_ids.index(zone)

    def minutes(self, pu: int, do: int, band: str) -> float:
        return float(self.bands[band]["minutes"][self._pos(pu), self._pos(do)])

    def miles(self, pu: int, do: int, band: str) -> float:
        return float(self.bands[band]["miles"][self._pos(pu), self._pos(do)])

    def matrix(self, band: str) -> np.ndarray:
        """n x n minutes, rows/cols ordered as `zone_ids`."""
        return self.bands[band]["minutes"]

    def miles_matrix(self, band: str) -> np.ndarray:
        return self.bands[band]["miles"]

    def level_of(self, pu: int, do: int, band: str) -> str:
        return LEVELS[int(self.bands[band]["level"][self._pos(pu), self._pos(do)])]
