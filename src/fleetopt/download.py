"""Fetch HVFHV Parquet files, the zone lookup, and zone centroids."""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import shapefile  # pyshp
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import unary_union

from fleetopt import constants as C

MIN_VALID_BYTES = 1_000_000


def month_url(month: str) -> str:
    return f"{C.TLC_TRIP_URL}/fhvhv_tripdata_{month}.parquet"


def _fetch(url: str, target: Path) -> Path:
    """Stream url to target via a .part file; skip if a plausible file already exists."""
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "fleetopt/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as out:
        shutil.copyfileobj(response, out)
    part.replace(target)
    return target


def download_month(month: str, destination: Path = C.RAW_DIR) -> Path:
    target = destination / f"fhvhv_tripdata_{month}.parquet"
    if target.exists() and target.stat().st_size < MIN_VALID_BYTES:
        target.unlink()  # truncated or error page from an earlier attempt
    return _fetch(month_url(month), target)


def build_zones_table(shapes_zip: Path, lookup_csv: Path) -> pd.DataFrame:
    """Zone centroids (lon/lat plus projected feet) joined to the TLC lookup."""
    extract_dir = shapes_zip.parent / "taxi_zones"
    with zipfile.ZipFile(shapes_zip) as zf:
        zf.extractall(extract_dir)
    shp_path = next(extract_dir.rglob("*.shp"))
    to_lonlat = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)
    reader = shapefile.Reader(str(shp_path))
    geoms: dict[int, list] = {}
    for sr in reader.shapeRecords():
        loc_id = int(sr.record["LocationID"])
        geoms.setdefault(loc_id, []).append(shape(sr.shape.__geo_interface__))
    rows = []
    for loc_id, parts in geoms.items():
        centroid = unary_union(parts).centroid  # some zones have several polygon records
        lon, lat = to_lonlat.transform(centroid.x, centroid.y)
        rows.append(
            {"location_id": loc_id, "x_ft": centroid.x, "y_ft": centroid.y, "lon": lon, "lat": lat}
        )
    lookup = pd.read_csv(lookup_csv, keep_default_na=False).rename(
        columns={
            "LocationID": "location_id",
            "Borough": "borough",
            "Zone": "zone",
            "service_zone": "service_zone",
        }
    )
    return lookup.merge(pd.DataFrame(rows), on="location_id", how="left")


def download_zones(destination: Path = C.RAW_DIR) -> Path:
    lookup = _fetch(f"{C.TLC_MISC_URL}/{C.ZONE_LOOKUP_FILE}", destination / C.ZONE_LOOKUP_FILE)
    shapes = _fetch(f"{C.TLC_MISC_URL}/{C.ZONE_SHAPES_FILE}", destination / C.ZONE_SHAPES_FILE)
    zones = build_zones_table(shapes, lookup)
    C.ZONES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    zones.to_parquet(C.ZONES_PARQUET, index=False)
    return C.ZONES_PARQUET


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", action="store_true", help="download only the dev month")
    args = parser.parse_args()
    months = C.DEV_MONTHS if args.dev else C.ALL_MONTHS
    print(f"Zones: {download_zones()}")
    for month in months:
        path = download_month(month)
        print(f"Ready: {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
