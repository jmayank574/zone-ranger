"""DuckDB connections. Every connection spills to a directory under the repo (never %TEMP%)."""

from pathlib import Path

import duckdb

from fleetopt import constants as C


def connect(path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """In-memory connection by default; pass C.DUCKDB_PATH for the persistent database."""
    C.DUCKDB_TMP_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path) if path else ":memory:")
    con.execute(f"SET temp_directory = '{C.DUCKDB_TMP_DIR.as_posix()}'")
    return con
