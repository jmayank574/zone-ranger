"""Zone lookups backed by data/zones.parquet (built by zone-ranger-download)."""

import numpy as np
import pandas as pd

from zone_ranger import constants as C


def load_zones() -> pd.DataFrame:
    return pd.read_parquet(C.ZONES_PARQUET)


def geofence_zone_ids(boroughs: tuple[str, ...] = C.GEOFENCE_BOROUGHS) -> list[int]:
    """Sorted LocationIDs of the service area (default: Manhattan, 69 zones)."""
    z = load_zones()
    ids = z[z.borough.isin(boroughs) & ~z.location_id.isin(C.INVALID_ZONES)].location_id
    return sorted(int(i) for i in ids)


def centroids(zone_ids: list[int]) -> pd.DataFrame:
    """lat/lon per zone, rows in the order of `zone_ids`."""
    z = load_zones().set_index("location_id").loc[zone_ids, ["lat", "lon"]]
    return z.reset_index(names="zone")


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (vectorized over numpy arrays / pandas Series)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (
        np.sin((p2 - p1) / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))
