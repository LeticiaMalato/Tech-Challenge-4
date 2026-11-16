"""Simulate two drift scenarios on the production dataset (explicit, reproducible).

1. Data drift (inputs change):
   - wages rise by a random 0%-40% per client; prices rise 20% for everyone;
   - a share of "Working" clients become "Unemployed" (income type and employment
     code change together). Their labels follow the default rate of unemployed
     clients in the reference: this is covariate shift into a region poorly covered
     by the training data (17 unemployed clients), not a change in P(Y|X).
2. Concept drift (inputs unchanged, risk changes):
   - clients with a high payment burden (annuity/income) default twice as often.
     No feature changes, so an input-drift detector should see nothing.

Label changes keep existing defaulters and turn part of the payers into
defaulters, preserving part of the original risk structure.
"""

import json

import numpy as np
import pandas as pd

from credit_monitoring import config


def raise_default_rate(
    df: pd.DataFrame, rows: pd.Index, target_rate: float, rng: np.random.Generator
) -> int:
    """Turn payers in `rows` into defaulters until the group reaches `target_rate`."""
    labels = df.loc[rows, config.TARGET_COLUMN]
    n_needed = round(float(target_rate) * len(rows)) - int(labels.sum())
    if n_needed <= 0:
        return 0
    payers = labels.index[labels == 0]
    converted = rng.choice(payers, size=min(n_needed, len(payers)), replace=False)
    df.loc[converted, config.TARGET_COLUMN] = 1
    return len(converted)


# ---------- Scenario 1: data drift ----------


def simulate_inflation(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Multiply each client's income by its own random wage adjustment."""
    factor = rng.uniform(
        1 + config.INFLATION_MIN, 1 + config.INFLATION_MAX, size=len(df)
    )
    df["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"] * factor
    return df


def simulate_price_inflation(df: pd.DataFrame) -> pd.DataFrame:
    """Prices rise by the same index for everyone: goods, loan amount and annuity."""
    for col in config.PRICE_COLUMNS:
        df[col] = df[col] * (1 + config.PRICE_INFLATION)
    return df


def simulate_unemployment(
    df: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.Index]:
    """Move a share of workers to unemployment; their labels follow the reference rate."""
    workers = df.index[df["NAME_INCOME_TYPE"] == "Working"]
    n_new = int(len(workers) * config.UNEMPLOYMENT_SHARE)
    newly_unemployed = pd.Index(rng.choice(workers, size=n_new, replace=False))

    df.loc[newly_unemployed, "NAME_INCOME_TYPE"] = "Unemployed"
    df.loc[newly_unemployed, "DAYS_EMPLOYED"] = config.DAYS_EMPLOYED_SENTINEL
    raise_default_rate(df, newly_unemployed, config.UNEMPLOYED_DEFAULT_RATE, rng)
    return df, newly_unemployed


def simulate_data_drift(production: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
    rng = np.random.default_rng(config.RANDOM_STATE)
    drifted = production.copy()
    drifted = simulate_inflation(drifted, rng)
    drifted = simulate_price_inflation(drifted)
    drifted, newly_unemployed = simulate_unemployment(drifted, rng)
    return drifted, newly_unemployed


# ---------- Scenario 2: concept drift ----------


def payment_burden(df: pd.DataFrame) -> pd.Series:
    return df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]


def simulate_concept_drift(
    production: pd.DataFrame, burden_limit: float
) -> tuple[pd.DataFrame, pd.Index]:
    """Same features; clients above the burden limit default more often."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    concept = production.copy()
    group = concept.index[payment_burden(concept) >= burden_limit]
    base_rate = concept.loc[group, config.TARGET_COLUMN].mean()
    raise_default_rate(
        concept, group, base_rate * config.CONCEPT_DEFAULT_MULTIPLIER, rng
    )
    return concept, group


if __name__ == "__main__":
    target = config.TARGET_COLUMN
    production = pd.read_csv(config.PRODUCTION_RAW_PATH)
    reference = pd.read_csv(config.REFERENCE_RAW_PATH)

    # Scenario 1
    drifted, newly_unemployed = simulate_data_drift(production)
    drifted.to_csv(config.PRODUCTION_DRIFTED_PATH, index=False)

    # Scenario 2: the burden limit is learned on the reference and frozen
    burden_limit = float(payment_burden(reference).quantile(config.COMMITMENT_QUANTILE))
    concept, burden_group = simulate_concept_drift(production, burden_limit)
    concept.to_csv(config.PRODUCTION_CONCEPT_PATH, index=False)

    summary = {
        "random_state": config.RANDOM_STATE,
        "rows": len(production),
        "data_drift": {
            "income_median_before": float(production["AMT_INCOME_TOTAL"].median()),
            "income_median_after": float(drifted["AMT_INCOME_TOTAL"].median()),
            "annuity_median_before": float(production["AMT_ANNUITY"].median()),
            "annuity_median_after": float(drifted["AMT_ANNUITY"].median()),
            "newly_unemployed": len(newly_unemployed),
            "default_rate_newly_unemployed_before": float(
                production.loc[newly_unemployed, target].mean()
            ),
            "default_rate_newly_unemployed_after": float(
                drifted.loc[newly_unemployed, target].mean()
            ),
            "default_rate_before": float(production[target].mean()),
            "default_rate_after": float(drifted[target].mean()),
        },
        "concept_drift": {
            "burden_limit": burden_limit,
            "group_size": len(burden_group),
            "group_default_rate_before": float(
                production.loc[burden_group, target].mean()
            ),
            "group_default_rate_after": float(concept.loc[burden_group, target].mean()),
            "default_rate_before": float(production[target].mean()),
            "default_rate_after": float(concept[target].mean()),
            "features_changed": bool(
                not concept.drop(columns=[target]).equals(
                    production.drop(columns=[target])
                )
            ),
        },
        "parameters": {
            "inflation_range": [config.INFLATION_MIN, config.INFLATION_MAX],
            "price_inflation": config.PRICE_INFLATION,
            "unemployment_share": config.UNEMPLOYMENT_SHARE,
            "unemployed_default_rate": config.UNEMPLOYED_DEFAULT_RATE,
            "commitment_quantile": config.COMMITMENT_QUANTILE,
            "concept_default_multiplier": config.CONCEPT_DEFAULT_MULTIPLIER,
        },
    }
    config.REPORTS_DIR.mkdir(exist_ok=True)
    config.SIMULATION_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
