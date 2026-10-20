"""Single source of configuration: months, splits, geofence, paths, seeds."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REPORTS_DIR = ROOT / "reports"
ARTIFACTS_DIR = ROOT / "artifacts"

TLC_TRIP_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
TLC_MISC_URL = "https://d37ci6vzurychx.cloudfront.net/misc"
ZONE_LOOKUP_FILE = "taxi_zone_lookup.csv"
ZONE_SHAPES_FILE = "taxi_zones.zip"
ZONES_PARQUET = DATA_DIR / "zones.parquet"

TRAIN_MONTHS = ("2025-01", "2025-02")
VAL_MONTHS = ("2025-03",)
TEST_MONTHS = ("2025-04",)
ALL_MONTHS = TRAIN_MONTHS + VAL_MONTHS + TEST_MONTHS
# --dev runs every stage on one month (and, downstream, a small sample).
DEV_MONTHS = ("2025-01",)

# TLC zones 264 ("Unknown") and 265 ("Outside NYC") carry no location information.
INVALID_ZONES = frozenset({264, 265})
SEED = 42

# Default service area: Manhattan zones, resolved from the zone lookup at runtime
# (see zones.manhattan_zone_ids). Override with an explicit list of LocationIDs.
GEOFENCE_BOROUGHS = ("Manhattan",)
DEMAND_SAMPLE_RATE = 0.05
