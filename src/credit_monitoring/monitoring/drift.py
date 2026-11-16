"""Statistical drift detection: reference vs production (control and drifted).

- Evidently produces the visual HTML report (PSI as the drift method).
- A summary table with PSI and KS per column is saved as JSON for monitoring.
Only information available on the decision day is compared: model features and
the model's risk score. The target is not used (labels arrive months later).
"""

import json

import joblib
import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

from credit_monitoring import config
from credit_monitoring.data.contract import validate_batch
from credit_monitoring.data.features import build_features
from credit_monitoring.modeling.train import prepare_features

DISCRETE = ["REGION_RATING_CLIENT_W_CITY", "HOUSING_INFO_MISSING"]
NUMERIC = [col for col in config.NUMERIC_FEATURES if col not in DISCRETE] + [
    "RISK_SCORE"
]
CATEGORICAL = config.CATEGORICAL_FEATURES + DISCRETE
EPS = 1e-6  # avoids log(0) when a bin is empty


def _psi(ref_pct: np.ndarray, cur_pct: np.ndarray) -> float:
    ref_pct = np.clip(ref_pct, EPS, None)
    cur_pct = np.clip(cur_pct, EPS, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_numeric(reference: pd.Series, current: pd.Series) -> float:
    """PSI with reference-decile bins plus one bin for missing values."""
    ref_all = reference.to_numpy(dtype=float)
    cur_all = current.to_numpy(dtype=float)
    ref, cur = ref_all[~np.isnan(ref_all)], cur_all[~np.isnan(cur_all)]

    quantiles = np.quantile(ref, np.linspace(0, 1, config.PSI_BINS + 1))
    inner_edges = np.unique(quantiles[1:-1])
    n_bins = len(inner_edges) + 1
    ref_counts = np.bincount(
        np.searchsorted(inner_edges, ref, side="right"), minlength=n_bins
    )
    cur_counts = np.bincount(
        np.searchsorted(inner_edges, cur, side="right"), minlength=n_bins
    )

    # Missing values get their own bin, so a change in the missing rate is visible
    ref_counts = np.append(ref_counts, np.isnan(ref_all).sum())
    cur_counts = np.append(cur_counts, np.isnan(cur_all).sum())
    return _psi(ref_counts / len(ref_all), cur_counts / len(cur_all))


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    """PSI over the categories seen in either dataset."""
    categories = sorted(set(reference) | set(current))
    ref_pct = reference.value_counts(normalize=True).reindex(categories, fill_value=0)
    cur_pct = current.value_counts(normalize=True).reindex(categories, fill_value=0)
    return _psi(ref_pct.to_numpy(), cur_pct.to_numpy())


def psi_level(psi: float) -> str:
    if psi < config.PSI_MODERATE:
        return "stable"
    if psi < config.PSI_SIGNIFICANT:
        return "moderate"
    return "significant"


def drift_table(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """PSI (decision metric) and KS (statistical context) for each monitored column."""
    rows = []
    for col in NUMERIC:
        psi = psi_numeric(reference[col], current[col])
        ks = ks_2samp(reference[col].dropna(), current[col].dropna())
        rows.append(
            {
                "column": col,
                "type": "numeric",
                "psi": psi,
                "psi_level": psi_level(psi),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
            }
        )
    for col in CATEGORICAL:
        psi = psi_categorical(reference[col], current[col])
        rows.append(
            {
                "column": col,
                "type": "categorical",
                "psi": psi,
                "psi_level": psi_level(psi),
                "ks_statistic": None,
                "ks_pvalue": None,
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, path) -> None:
    """Visual drift report (Evidently), using PSI as the drift method."""
    definition = DataDefinition(
        numerical_columns=NUMERIC, categorical_columns=CATEGORICAL
    )
    columns = NUMERIC + CATEGORICAL
    ref_ds = Dataset.from_pandas(reference[columns], data_definition=definition)
    cur_ds = Dataset.from_pandas(current[columns], data_definition=definition)
    report = Report([DataDriftPreset(method="psi")])
    snapshot = report.run(current_data=cur_ds, reference_data=ref_ds)
    snapshot.save_html(str(path))


def build_reference(model) -> pd.DataFrame:
    """Reference = the baseline TEST set (out-of-sample scores), same split as train.py."""
    valid, _, _ = validate_batch(pd.read_csv(config.REFERENCE_RAW_PATH))
    data = build_features(valid)
    _, reference = train_test_split(
        data,
        test_size=config.TEST_SIZE,
        stratify=data[config.TARGET_COLUMN],
        random_state=config.RANDOM_STATE,
    )
    reference = reference.copy()
    reference["RISK_SCORE"] = model.predict_proba(prepare_features(reference))[:, 1]
    reference[CATEGORICAL] = reference[CATEGORICAL].astype(str)
    return reference


if __name__ == "__main__":
    model = joblib.load(config.MODEL_PATH)
    reference = build_reference(model)

    batches = {
        "control": config.PREDICTIONS_CONTROL_PATH,
        "data_drift": config.PREDICTIONS_DRIFTED_PATH,
        "concept_drift": config.PREDICTIONS_CONCEPT_PATH,
    }
    results = {}
    for name, path in batches.items():
        current = pd.read_csv(path)
        current[CATEGORICAL] = current[CATEGORICAL].astype(str)

        table = drift_table(reference, current)
        results[name] = table.to_dict(orient="records")
        evidently_report(
            reference, current, config.REPORTS_DIR / f"drift_report_{name}.html"
        )

        print(f"\n=== {name} ===")
        print(table.round(4).to_string(index=False))

    config.DRIFT_SUMMARY_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSummary saved to {config.DRIFT_SUMMARY_PATH}")
