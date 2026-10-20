from fleetopt import constants as C
from fleetopt import db


def test_connect_sets_temp_directory_under_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "DUCKDB_TMP_DIR", tmp_path / "spill")
    con = db.connect()
    got = con.execute("SELECT current_setting('temp_directory')").fetchone()[0]
    assert got.replace("\\", "/") == (tmp_path / "spill").as_posix()
    assert (tmp_path / "spill").is_dir()
