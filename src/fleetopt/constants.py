"""Single source of configuration: months, splits, geofence, paths, seeds."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REPORTS_DIR = ROOT / "reports"
ARTIFACTS_DIR = ROOT / "artifacts"
DUCKDB_PATH = DATA_DIR / "fleetopt.duckdb"  # persistent DB used from Phase 1
DUCKDB_TMP_DIR = DATA_DIR / "duckdb_tmp"  # spill directory; keeps DuckDB off the C: drive

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
DEFAULT_FLEET_SIZE = 225  # config parameter; sweep {150, 225, 300} in Phase 5

# --- Phase 1: features, models, travel matrix ---
SLOT_MINUTES = 15
SLOTS_PER_DAY = 96
DEMAND_LAGS = (1, 4, 96, 672)  # slots: 15 min, 1 h, 1 day, 1 week
DEMAND_ROLLING = (4, 96)  # slots; means over the window ending `horizon` slots before t
DEMAND_HORIZON = 1  # slots ahead (1 = one step ahead)

# Window bounds are [start, end) on the naive local-time slot axis. Train starts 7 days in so the
# weekly lag exists. Test stops 30 min before the end of April: requests there can have pickups in
# May, which the April file does not contain (right-censoring).
SPLITS = {
    "train": ("2025-01-08", "2025-03-01"),
    "val": ("2025-03-01", "2025-04-01"),
    "test": ("2025-04-01", "2025-04-30 23:30"),
}
DEV_SPLITS = {
    "train": ("2025-01-08", "2025-01-15"),
    "val": ("2025-01-15", "2025-01-18"),
    "test": ("2025-01-18", "2025-01-22"),
}
GRID_BOUNDS = ("2025-01-01", "2025-05-01")
DEV_GRID_BOUNDS = ("2025-01-01", "2025-01-22")
DEV_DUCKDB_PATH = DATA_DIR / "fleetopt_dev.duckdb"
DEV_THIN_PERCENT = 1  # keep ~1% of trips in --dev (deterministic hash filter)
LOCAL_TZ = "America/New_York"  # raw timestamps are naive local time

# US federal holidays inside the data window (observed dates).
HOLIDAYS = ("2025-01-01", "2025-01-20", "2025-02-17")

# Time-of-day bands for the travel matrix: (name, first hour, last hour exclusive).
TIME_BANDS = (("overnight", 0, 6), ("am_peak", 6, 10), ("midday", 10, 16), ("pm_evening", 16, 24))
TRAVEL_MIN_TRIPS = 20
