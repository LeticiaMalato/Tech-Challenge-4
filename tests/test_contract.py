"""Tests for the data contract.

Each test builds its own small DataFrame, so no dataset file is needed.
"""

import pandas as pd

from credit_monitoring import config
from credit_monitoring.data.contract import validate_batch


def valid_row(**overrides) -> pd.DataFrame:
    """A single valid record; keyword arguments change specific fields."""
    row = {
        "SK_ID_CURR": 1,
        "DAYS_BIRTH": -12000,  # ~33 years
        "AMT_INCOME_TOTAL": 150000.0,
        "AMT_ANNUITY": 20000.0,
        "FLAG_OWN_CAR": "N",
        "OWN_CAR_AGE": None,
        "DAYS_EMPLOYED": -1500,
        "NAME_INCOME_TYPE": "Working",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def assert_blocked_only_by(blocked: pd.DataFrame, rule: str) -> None:
    """The record must be blocked, and every failure must come from the given rule."""
    assert not blocked.empty, f"expected a block by '{rule}', but nothing was blocked"
    assert blocked["check"].str.startswith(rule).all(), blocked["check"].tolist()


#  Blocking rules: one error per test


def test_valid_record_passes():
    valid, blocked, alerts = validate_batch(valid_row())
    assert len(valid) == 1
    assert blocked.empty
    assert alerts.empty


def test_underage_client_is_blocked():
    valid, blocked, _ = validate_batch(valid_row(DAYS_BIRTH=-5844))  # ~16 years
    assert valid.empty
    assert_blocked_only_by(blocked, "in_range")


def test_zero_income_is_blocked():
    valid, blocked, _ = validate_batch(valid_row(AMT_INCOME_TOTAL=0.0))
    assert valid.empty
    assert_blocked_only_by(blocked, "greater_than")


def test_missing_income_is_blocked():
    valid, blocked, _ = validate_batch(valid_row(AMT_INCOME_TOTAL=None))
    assert valid.empty
    assert_blocked_only_by(blocked, "not_nullable")


def test_car_owner_without_car_age_is_blocked():
    valid, blocked, _ = validate_batch(valid_row(FLAG_OWN_CAR="Y", OWN_CAR_AGE=None))
    assert valid.empty
    assert_blocked_only_by(blocked, "car_consistency")


def test_car_age_without_car_is_blocked():
    valid, blocked, _ = validate_batch(valid_row(FLAG_OWN_CAR="N", OWN_CAR_AGE=5.0))
    assert valid.empty
    assert_blocked_only_by(blocked, "car_consistency")


def test_employment_code_for_worker_is_blocked():
    row = valid_row(
        DAYS_EMPLOYED=config.DAYS_EMPLOYED_SENTINEL, NAME_INCOME_TYPE="Working"
    )
    valid, blocked, _ = validate_batch(row)
    assert valid.empty
    assert_blocked_only_by(blocked, "employment_consistency")


def test_positive_employment_days_are_blocked():
    valid, blocked, _ = validate_batch(valid_row(DAYS_EMPLOYED=100))
    assert valid.empty
    assert_blocked_only_by(blocked, "employment_consistency")


def test_duplicate_id_blocks_only_the_repetition():
    first = valid_row(SK_ID_CURR=1)
    second = valid_row(
        SK_ID_CURR=1, AMT_INCOME_TOTAL=200000.0
    )  # same ID, different record
    batch = pd.concat([first, second], ignore_index=True)
    valid, blocked, _ = validate_batch(batch)
    assert len(valid) == 1
    assert_blocked_only_by(blocked, "field_uniqueness")


def test_duplicate_record_blocks_only_the_repetition():
    first = valid_row(SK_ID_CURR=1)
    second = valid_row(SK_ID_CURR=2)  # same record, different ID
    batch = pd.concat([first, second], ignore_index=True)
    valid, blocked, _ = validate_batch(batch)
    assert len(valid) == 1
    assert_blocked_only_by(blocked, "no_duplicate_records")


#  Records that must NOT be blocked


def test_employment_code_for_pensioner_passes():
    row = valid_row(
        DAYS_EMPLOYED=config.DAYS_EMPLOYED_SENTINEL, NAME_INCOME_TYPE="Pensioner"
    )
    valid, blocked, _ = validate_batch(row)
    assert len(valid) == 1
    assert blocked.empty


#  Alert rules: the record passes, but is flagged


def test_old_car_passes_with_alert():
    row = valid_row(FLAG_OWN_CAR="Y", OWN_CAR_AGE=config.CAR_AGE_ALERT + 4)
    valid, blocked, alerts = validate_batch(row)
    assert len(valid) == 1
    assert blocked.empty
    assert (alerts["check"] == "car_age_above_limit").any()


def test_high_income_passes_with_alert():
    valid, blocked, alerts = validate_batch(
        valid_row(AMT_INCOME_TOTAL=config.INCOME_ALERT_THRESHOLD * 10)
    )
    assert len(valid) == 1
    assert blocked.empty
    assert (alerts["check"] == "income_above_threshold").any()


def test_annuity_above_income_passes_with_alert():
    valid, blocked, alerts = validate_batch(valid_row(AMT_ANNUITY=200000.0))
    assert len(valid) == 1
    assert blocked.empty
    assert (alerts["check"] == "annuity_above_income").any()
