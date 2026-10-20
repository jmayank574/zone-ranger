import shapefile
from shapely.geometry import box

from zone_ranger import download as D


def test_month_url():
    assert D.month_url("2025-01").endswith("/fhvhv_tripdata_2025-01.parquet")


def test_fetch_skips_existing_file(tmp_path, monkeypatch):
    target = tmp_path / "x.csv"
    target.write_text("data")
    monkeypatch.setattr(D.urllib.request, "urlopen", lambda *a, **k: 1 / 0)
    assert D._fetch("http://unused", target) == target


def test_truncated_month_file_is_removed_and_refetched(tmp_path, monkeypatch):
    bad = tmp_path / "fhvhv_tripdata_2025-01.parquet"
    bad.write_bytes(b"oops")
    calls = []
    monkeypatch.setattr(D, "_fetch", lambda url, target: calls.append(url) or target)
    D.download_month("2025-01", tmp_path)
    assert calls and not bad.exists()


def test_build_zones_table_merges_multipart_zone(tmp_path):
    shp = tmp_path / "z" / "z.shp"
    shp.parent.mkdir()
    with shapefile.Writer(str(shp)) as w:
        w.field("LocationID", "N")
        for loc, geom in [(1, box(0, 0, 10, 10)), (1, box(20, 0, 30, 10)), (2, box(0, 20, 10, 30))]:
            w.poly([list(geom.exterior.coords)])
            w.record(loc)
    (tmp_path / "z" / "z.prj").write_text("")
    import zipfile

    zpath = tmp_path / "zones.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for p in (shp.parent).iterdir():
            zf.write(p, p.name)
    csv = tmp_path / "lookup.csv"
    csv.write_text("LocationID,Borough,Zone,service_zone\n1,Manhattan,A,Y\n2,N/A,B,N/A\n")
    out = D.build_zones_table(zpath, csv)
    assert len(out) == 2
    assert out.loc[out.location_id == 1, "x_ft"].iloc[0] == 15  # centroid of both parts
    assert out.loc[out.location_id == 2, "borough"].iloc[0] == "N/A"  # not parsed as NaN
