import duckdb

from zone_ranger import cleaning, eda


def test_fare_rule_flags_not_drops():
    assert "fare" not in " ".join(cleaning.RULES)  # fares are never a drop rule
    con = duckdb.connect()
    values = "(NULL::DOUBLE), (0.0), (-1.0), (0.01), (499.99), (500.0)"
    got = con.execute(
        f"SELECT ({cleaning.FARE_INVALID}) FROM (VALUES {values}) t(base_passenger_fare)"
    ).fetchall()
    assert [bool(r[0]) for r in got] == [True, True, True, False, False, True]


def test_eda_uses_shared_rules():
    assert eda.RULES is cleaning.RULES
    assert eda.WAIT_ANOMALIES is cleaning.WAIT_ANOMALIES
