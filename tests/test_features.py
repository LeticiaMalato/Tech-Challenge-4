"""Tests for the feature transformations."""

import numpy as np
import pandas as pd
import pytest

from credit_monitoring import config
from credit_monitoring.data.features import build_features


def raw_rows() -> pd.DataFrame:
    """Two raw records with the columns used by the transformations."""
    return pd.DataFrame(
        {
            "DAYS_BIRTH": [-12000, -20000],
            "OWN_CAR_AGE": [7.0, np.nan],  # car aged 7 / no car
            "DAYS_EMPLOYED": [
                -800,
                config.DAYS_EMPLOYED_SENTINEL,
            ],  # ~2.2 years / not employed
            "APARTMENTS_AVG": [0.1, np.nan],  # housing info present / missing
            "APARTMENTS_MODE": [np.nan, np.nan],
        }
    )


def test_age_is_positive_years():
    features = build_features(raw_rows())
    assert features["AGE_YEARS"].iloc[0] == 12000 / config.DAYS_PER_YEAR


def test_car_age_groups_and_no_car():
    features = build_features(raw_rows())
    assert features["CAR_AGE_GROUP"].tolist() == ["5-10", "NO_CAR"]


def test_employment_groups_and_sentinel():
    features = build_features(raw_rows())
    assert features["EMPLOYMENT_GROUP"].tolist() == ["1-3", "NOT_EMPLOYED"]


def test_housing_indicator_replaces_housing_columns():
    features = build_features(raw_rows())
    assert features["HOUSING_INFO_MISSING"].tolist() == [0, 1]
    assert not any(col.endswith(config.HOUSING_SUFFIXES) for col in features.columns)


def test_input_is_not_modified():
    raw = raw_rows()
    original = raw.copy()
    build_features(raw)
    pd.testing.assert_frame_equal(raw, original)


def test_missing_housing_columns_raise():
    raw = raw_rows().drop(columns=["APARTMENTS_AVG", "APARTMENTS_MODE"])
    with pytest.raises(ValueError, match="housing"):
        build_features(raw)
