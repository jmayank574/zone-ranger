from fleetopt import constants as C


def test_splits_are_strictly_ordered_and_disjoint():
    months = C.TRAIN_MONTHS + C.VAL_MONTHS + C.TEST_MONTHS
    assert list(months) == sorted(set(months))
    assert max(C.TRAIN_MONTHS) < min(C.VAL_MONTHS) < min(C.TEST_MONTHS)


def test_dev_month_is_in_train_window():
    assert set(C.DEV_MONTHS) <= set(C.TRAIN_MONTHS)


def test_invalid_zones_are_264_and_265():
    assert {264, 265} == C.INVALID_ZONES


def test_all_paths_live_under_repo_root():
    for p in (
        C.DATA_DIR,
        C.RAW_DIR,
        C.REPORTS_DIR,
        C.ARTIFACTS_DIR,
        C.DUCKDB_PATH,
        C.DUCKDB_TMP_DIR,
        C.ZONES_PARQUET,
    ):
        assert p.is_relative_to(C.ROOT), p
