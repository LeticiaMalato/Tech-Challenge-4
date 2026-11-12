"""Feature transformations decided in the reference EDA.

All transformations are fixed rules (no statistics learned from the data),
so they are applied in the same way to reference and production.
"""

import pandas as pd

from credit_monitoring import config


def add_housing_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the 47 housing columns with a single 'all missing' indicator."""
    housing_cols = [col for col in df.columns if col.endswith(config.HOUSING_SUFFIXES)]
    df["HOUSING_INFO_MISSING"] = df[housing_cols].isnull().all(axis=1).astype(int)
    return df.drop(columns=housing_cols)


def add_car_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Group car age into fixed bins; clients without a car get NO_CAR."""
    groups = pd.cut(
        df["OWN_CAR_AGE"],
        bins=config.CAR_AGE_BINS,
        labels=config.CAR_AGE_LABELS,
        include_lowest=True,
    )
    df["CAR_AGE_GROUP"] = groups.cat.add_categories("NO_CAR").fillna("NO_CAR")
    return df


def add_age_years(df: pd.DataFrame) -> pd.DataFrame:
    """Convert DAYS_BIRTH (negative days) into age in years."""
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / config.DAYS_PER_YEAR
    return df


def add_employment_group(df: pd.DataFrame) -> pd.DataFrame:
    """Group job tenure into fixed bins; the sentinel code becomes NOT_EMPLOYED."""
    is_sentinel = df["DAYS_EMPLOYED"] == config.DAYS_EMPLOYED_SENTINEL
    years = (-df["DAYS_EMPLOYED"] / config.DAYS_PER_YEAR).where(~is_sentinel)
    groups = pd.cut(
        years,
        bins=config.EMPLOYMENT_BINS,
        labels=config.EMPLOYMENT_LABELS,
        include_lowest=True,
    )
    df["EMPLOYMENT_GROUP"] = groups.cat.add_categories("NOT_EMPLOYED").fillna(
        "NOT_EMPLOYED"
    )
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all transformations to a validated batch (the input is not modified)."""
    df = df.copy()
    df = add_housing_indicator(df)
    df = add_car_age_group(df)
    df = add_age_years(df)
    df = add_employment_group(df)
    return df


if __name__ == "__main__":
    reference = pd.read_csv(config.REFERENCE_CLEAN_PATH)
    features = build_features(reference)

    print(f"Shape: {reference.shape} -> {features.shape}")
    for col in ["HOUSING_INFO_MISSING", "CAR_AGE_GROUP", "EMPLOYMENT_GROUP"]:
        print(f"\n{col}:")
        print(features[col].value_counts().to_string())
    print("\nAGE_YEARS:")
    print(features["AGE_YEARS"].describe().round(1).to_string())
