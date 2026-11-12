"""Data contract for raw credit application data (Pandera)."""

import pandas as pd
import pandera.pandas as pa

from credit_monitoring import config

# Age limits in DAYS_BIRTH (negative days): the oldest age gives the smallest value
MIN_DAYS_BIRTH = -config.MAX_AGE * config.DAYS_PER_YEAR
MAX_DAYS_BIRTH = -config.MIN_AGE * config.DAYS_PER_YEAR
NOT_EMPLOYED_INCOME_TYPES = ["Pensioner", "Unemployed"]


#  Dataframe-level rules (more than one column)


def no_duplicate_records(df: pd.DataFrame) -> pd.Series:
    """Rule 3: a record must not repeat another one under a different ID."""
    return ~df.drop(columns=[config.ID_COLUMN]).duplicated(keep="first")


def car_consistency(df: pd.DataFrame) -> pd.Series:
    """Rule 5: car owners must have a car age; non-owners must not."""
    return (df["FLAG_OWN_CAR"] == "Y") == df["OWN_CAR_AGE"].notna()


def employment_consistency(df: pd.DataFrame) -> pd.Series:
    """Rule 6: the sentinel code is only valid for pensioners and unemployed;
    any other value must be a non-positive number of days."""
    is_code = df["DAYS_EMPLOYED"] == config.DAYS_EMPLOYED_SENTINEL
    allowed_type = df["NAME_INCOME_TYPE"].isin(NOT_EMPLOYED_INCOME_TYPES)
    return (is_code & allowed_type) | (~is_code & (df["DAYS_EMPLOYED"] <= 0))


def annuity_within_income(df: pd.DataFrame) -> pd.Series:
    """Rule 8 (alert): the annuity should not exceed the total income."""
    return df["AMT_ANNUITY"].isna() | (df["AMT_ANNUITY"] <= df["AMT_INCOME_TOTAL"])


#  Blocking rules: a failing record is rejected

blocking_schema = pa.DataFrameSchema(
    columns={
        # Rule 1: age between MIN_AGE and MAX_AGE
        "DAYS_BIRTH": pa.Column(
            nullable=False,
            checks=pa.Check.in_range(MIN_DAYS_BIRTH, MAX_DAYS_BIRTH),
        ),
        # Rule 2: unique, non-null ID; only the repeated occurrence is reported
        config.ID_COLUMN: pa.Column(
            nullable=False, unique=True, report_duplicates="exclude_first"
        ),
        # Rule 4: income present and positive
        "AMT_INCOME_TOTAL": pa.Column(nullable=False, checks=pa.Check.gt(0)),
        # Columns required by the dataframe-level rules
        "FLAG_OWN_CAR": pa.Column(nullable=False, checks=pa.Check.isin(["Y", "N"])),
        "OWN_CAR_AGE": pa.Column(nullable=True),
        "DAYS_EMPLOYED": pa.Column(nullable=False),
        "NAME_INCOME_TYPE": pa.Column(nullable=False),
    },
    checks=[
        pa.Check(no_duplicate_records, name="no_duplicate_records"),
        pa.Check(car_consistency, name="car_consistency"),
        pa.Check(employment_consistency, name="employment_consistency"),
    ],
    strict=False,
)

#  Alert rules: a failing record passes, but is flagged

alert_schema = pa.DataFrameSchema(
    columns={
        # Rule 7: income above the reference p99.9 (nulls are ignored)
        "AMT_INCOME_TOTAL": pa.Column(
            nullable=True,
            checks=pa.Check(
                lambda s: s <= config.INCOME_ALERT_THRESHOLD,
                name="income_above_threshold",
            ),
        ),
        # Rule 9: car age above the suspicious-peak limit (no car = null, ignored)
        "OWN_CAR_AGE": pa.Column(
            nullable=True,
            checks=pa.Check(
                lambda s: s <= config.CAR_AGE_ALERT,
                name="car_age_above_limit",
            ),
        ),
        "AMT_ANNUITY": pa.Column(nullable=True),
    },
    checks=[pa.Check(annuity_within_income, name="annuity_above_income")],
    strict=False,
)


def _failure_cases(schema: pa.DataFrameSchema, df: pd.DataFrame) -> pd.DataFrame:
    """One row per (record, rule) that failed the schema."""
    try:
        schema.validate(df, lazy=True)
        return pd.DataFrame(columns=["index", "check"])
    except pa.errors.SchemaErrors as err:
        cases = err.failure_cases.dropna(subset=["index"])
        # Dataframe-level rules report every column of a failing row: keep one line per rule
        return cases.drop_duplicates(subset=["index", "check"])[["index", "check"]]


def validate_batch(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate a raw batch.

    Returns the valid rows, the blocked cases and the alert cases.
    """
    blocked = _failure_cases(blocking_schema, df)
    alerts = _failure_cases(alert_schema, df)
    valid = df.drop(index=blocked["index"].unique())
    return valid, blocked, alerts


# Invalid batch


def build_invalid_batch(reference: pd.DataFrame) -> pd.DataFrame:
    """Copies of valid rows, each with exactly one error. Index labels name the case."""
    # Base rows that are valid and trigger no alert, so each error is isolated
    pool = reference[
        (reference["NAME_INCOME_TYPE"] == "Working")
        & (reference["FLAG_OWN_CAR"] == "Y")
        & (reference["OWN_CAR_AGE"] <= config.CAR_AGE_ALERT)
        & (reference["AMT_INCOME_TOTAL"] <= config.INCOME_ALERT_THRESHOLD)
        & (reference["AMT_ANNUITY"] <= reference["AMT_INCOME_TOTAL"])
    ]

    def row(i: int) -> pd.DataFrame:
        return pool.iloc[[i]].copy()

    valid_1, valid_2 = row(0), row(1)

    age_under_18 = row(2)
    age_under_18["DAYS_BIRTH"] = int(-16 * config.DAYS_PER_YEAR)

    duplicate_id = row(3)
    duplicate_id[config.ID_COLUMN] = valid_1[config.ID_COLUMN].iloc[0]

    duplicate_record = valid_2.copy()
    duplicate_record[config.ID_COLUMN] = reference[config.ID_COLUMN].max() + 1

    zero_income = row(4)
    zero_income["AMT_INCOME_TOTAL"] = 0

    car_inconsistent = row(5)
    car_inconsistent["OWN_CAR_AGE"] = None

    employment_inconsistent = row(6)
    employment_inconsistent["DAYS_EMPLOYED"] = config.DAYS_EMPLOYED_SENTINEL

    car_age_alert = row(7)
    car_age_alert["OWN_CAR_AGE"] = 64

    batch = pd.concat(
        [
            valid_1,
            valid_2,
            age_under_18,
            duplicate_id,
            duplicate_record,
            zero_income,
            car_inconsistent,
            employment_inconsistent,
            car_age_alert,
        ]
    )
    batch.index = [
        "valid_1",
        "valid_2",
        "age_under_18",
        "duplicate_id",
        "duplicate_record",
        "zero_income",
        "car_inconsistent",
        "employment_inconsistent",
        "car_age_alert",
    ]
    return batch


if __name__ == "__main__":
    reference = pd.read_csv(config.REFERENCE_RAW_PATH)

    # 1. Reference: the clean dataset is defined as the rows that pass the contract
    valid, blocked, alerts = validate_batch(reference)
    print(f"Reference: {len(reference)} rows | {len(reference) - len(valid)} blocked")
    print("Blocked by rule:")
    print(blocked.groupby("check")["index"].nunique().to_string())
    print("Alerts by rule:")
    print(alerts.groupby("check")["index"].nunique().to_string())
    valid.to_csv(config.REFERENCE_CLEAN_PATH, index=False)

    # 2. Invalid batch: each row must be blocked by exactly one rule
    batch = build_invalid_batch(reference)
    valid_batch, blocked_batch, alerts_batch = validate_batch(batch)
    print("\nInvalid batch - blocked:")
    print(blocked_batch.to_string(index=False))
    print("Accepted:", list(valid_batch.index))
    print("Alerts:")
    print(alerts_batch.to_string(index=False))
