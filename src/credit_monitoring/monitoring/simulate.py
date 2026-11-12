"""Simulate an economic shift on the production dataset.

The scenario is explicit and reproducible (fixed seed):
- Inflation: declared income rises by a random 0%-40% per client (+20% on average).
- Unemployment: a share of "Working" clients become "Unemployed" (income type and
  employment code change together, so the data contract accepts them).
- Simulated concept drift: newly unemployed clients get new labels drawn from the
  default rate of unemployed clients in the reference.
"""

import json

import numpy as np
import pandas as pd

from credit_monitoring import config


def simulate_inflation(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Multiply each client's income by its own random inflation factor."""
    factor = rng.uniform(
        1 + config.INFLATION_MIN, 1 + config.INFLATION_MAX, size=len(df)
    )
    df["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"] * factor
    return df


def simulate_unemployment(
    df: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.Index]:
    """Move a share of workers to unemployment and redraw their labels."""
    workers = df.index[df["NAME_INCOME_TYPE"] == "Working"]
    n_new = int(len(workers) * config.UNEMPLOYMENT_SHARE)
    newly_unemployed = pd.Index(rng.choice(workers, size=n_new, replace=False))

    # Data drift: both columns change together, keeping the record consistent
    df.loc[newly_unemployed, "NAME_INCOME_TYPE"] = "Unemployed"
    df.loc[newly_unemployed, "DAYS_EMPLOYED"] = config.DAYS_EMPLOYED_SENTINEL

    # Simulated concept drift: same profile, higher risk than the model learned
    new_labels = (rng.uniform(size=n_new) < config.UNEMPLOYED_DEFAULT_RATE).astype(int)
    df.loc[newly_unemployed, config.TARGET_COLUMN] = new_labels
    return df, newly_unemployed


def simulate_drift(production: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
    """Apply the full scenario to a copy of the production data."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    drifted = production.copy()
    drifted = simulate_inflation(drifted, rng)
    drifted, newly_unemployed = simulate_unemployment(drifted, rng)
    return drifted, newly_unemployed


if __name__ == "__main__":
    production = pd.read_csv(config.PRODUCTION_RAW_PATH)
    drifted, newly_unemployed = simulate_drift(production)
    drifted.to_csv(config.PRODUCTION_DRIFTED_PATH, index=False)

    target = config.TARGET_COLUMN
    summary = {
        "random_state": config.RANDOM_STATE,
        "rows": len(production),
        "income_median_before": float(production["AMT_INCOME_TOTAL"].median()),
        "income_median_after": float(drifted["AMT_INCOME_TOTAL"].median()),
        "unemployed_share_before": float(
            (production["NAME_INCOME_TYPE"] == "Unemployed").mean()
        ),
        "unemployed_share_after": float(
            (drifted["NAME_INCOME_TYPE"] == "Unemployed").mean()
        ),
        "newly_unemployed": len(newly_unemployed),
        "default_rate_before": float(production[target].mean()),
        "default_rate_after": float(drifted[target].mean()),
        "default_rate_newly_unemployed_before": float(
            production.loc[newly_unemployed, target].mean()
        ),
        "default_rate_newly_unemployed_after": float(
            drifted.loc[newly_unemployed, target].mean()
        ),
        "parameters": {
            "inflation_range": [config.INFLATION_MIN, config.INFLATION_MAX],
            "unemployment_share": config.UNEMPLOYMENT_SHARE,
            "unemployed_default_rate": config.UNEMPLOYED_DEFAULT_RATE,
        },
    }
    config.REPORTS_DIR.mkdir(exist_ok=True)
    config.SIMULATION_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
