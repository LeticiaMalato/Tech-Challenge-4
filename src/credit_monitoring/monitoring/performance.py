"""Label-based performance monitoring.

Input drift (PSI) cannot see concept drift: inputs and scores stay the same.
Only labels reveal it, and in credit they arrive months after the decision.
This report compares the scored batches once labels are available:
- calibration gap = observed default rate - mean predicted risk
- default rate among approved clients, ROC-AUC, PR-AUC normalized by prevalence
- calibration by score decile (deciles fixed on the control batch)
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from credit_monitoring import config

BATCHES = {
    "control": config.PREDICTIONS_CONTROL_PATH,
    "data_drift": config.PREDICTIONS_DRIFTED_PATH,
    "concept_drift": config.PREDICTIONS_CONCEPT_PATH,
}


def performance(df: pd.DataFrame) -> dict:
    """Performance and calibration of one scored batch with labels."""
    y = df[config.TARGET_COLUMN].to_numpy()
    proba = df["RISK_SCORE"].to_numpy()
    approved = df["APPROVED"].to_numpy(dtype=bool)
    default_rate = float(y.mean())
    mean_risk = float(proba.mean())
    pr_auc = float(average_precision_score(y, proba))
    return {
        "rows_scored": len(df),
        "default_rate": default_rate,
        "mean_risk_score": mean_risk,
        "calibration_gap": default_rate
        - mean_risk,  # > 0: the model underestimates risk
        "default_rate_approved": float(y[approved].mean()),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": pr_auc,
        "pr_auc_lift": pr_auc / default_rate,  # comparable across different prevalences
    }


def calibration_by_decile(df: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    """Mean predicted risk vs observed default rate, per fixed score band."""
    bands = pd.cut(df["RISK_SCORE"], bins=edges, include_lowest=True)
    return df.groupby(bands, observed=False).agg(
        clients=(config.TARGET_COLUMN, "size"),
        predicted=("RISK_SCORE", "mean"),
        observed=(config.TARGET_COLUMN, "mean"),
    )


def build_html(summary: pd.DataFrame, deciles: dict[str, pd.DataFrame]) -> str:
    """A self-contained HTML page with the comparison tables."""
    sections = [
        "<h1>Label-based performance monitoring</h1>",
        (
            "<p>Labels arrive months after the decision. Input-drift monitoring (PSI) "
            "cannot detect concept drift, because inputs and scores do not change. "
            "Calibration gap = observed default rate - mean predicted risk.</p>"
        ),
        "<h2>Summary by batch</h2>",
        summary.round(4).to_html(),
        "<h2>Calibration by score band (bands fixed on the control batch)</h2>",
    ]
    for name, table in deciles.items():
        sections += [f"<h3>{name}</h3>", table.round(4).to_html()]
    style = (
        "<style>body{font-family:sans-serif;margin:2rem;max-width:960px}"
        "table{border-collapse:collapse;margin-bottom:1.5rem}"
        "td,th{border:1px solid #ccc;padding:4px 8px;text-align:right}</style>"
    )
    return (
        f"<!doctype html><html><head><meta charset='utf-8'>{style}</head><body>"
        + "".join(sections)
        + "</body></html>"
    )


if __name__ == "__main__":
    batches = {name: pd.read_csv(path) for name, path in BATCHES.items()}

    summary = pd.DataFrame({name: performance(df) for name, df in batches.items()}).T

    # Score bands from the control batch deciles, then frozen for every batch
    control_scores = batches["control"]["RISK_SCORE"]
    inner = np.unique(np.quantile(control_scores, np.linspace(0, 1, 11))[1:-1])
    edges = np.concatenate([[-np.inf], inner, [np.inf]])
    deciles = {name: calibration_by_decile(df, edges) for name, df in batches.items()}

    config.PERFORMANCE_SUMMARY_PATH.write_text(
        json.dumps(summary.to_dict(orient="index"), indent=2)
    )
    config.PERFORMANCE_REPORT_PATH.write_text(
        build_html(summary, deciles), encoding="utf-8"
    )

    print(summary.round(4).to_string())
    print(f"\nReport saved to {config.PERFORMANCE_REPORT_PATH}")
