"""Label-based performance monitoring.

Input drift (PSI) cannot see concept drift: inputs and scores stay the same.
Only labels reveal it, and in credit they arrive months after the decision.

Comparison base: the baseline TEST set ("reference"), the same base as drift.py.
Metrics per batch: calibration gap (observed default - mean predicted risk),
default rate among approved clients, ROC-AUC and PR-AUC normalized by prevalence.
Segments (diagnosis, not detection): newly unemployed and high payment burden.
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


def segment_calibration(df: pd.DataFrame, burden_limit: float) -> pd.DataFrame:
    """Calibration gap inside the segments affected by each scenario (diagnosis only)."""
    segments = {
        "newly_unemployed": df["NAME_INCOME_TYPE"] == "Unemployed",
        "high_payment_burden": (df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"])
        >= burden_limit,
    }
    rows = {}
    for name, mask in segments.items():
        part = df[mask]
        observed = part[config.TARGET_COLUMN].mean() if len(part) else np.nan
        predicted = part["RISK_SCORE"].mean() if len(part) else np.nan
        rows[name] = {
            "clients": len(part),
            "observed_default": observed,
            "predicted_risk": predicted,
            "calibration_gap": observed - predicted,
        }
    return pd.DataFrame(rows).T


def calibration_by_band(df: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    """Mean predicted risk vs observed default rate, per fixed score band."""
    bands = pd.cut(df["RISK_SCORE"], bins=edges, include_lowest=True)
    return df.groupby(bands, observed=False).agg(
        clients=(config.TARGET_COLUMN, "size"),
        predicted=("RISK_SCORE", "mean"),
        observed=(config.TARGET_COLUMN, "mean"),
    )


def verdict(gap: float) -> str:
    if abs(gap) < config.CALIBRATION_GAP_ALERT:
        return "calibrated"
    return (
        "the model UNDERESTIMATES risk" if gap > 0 else "the model OVERESTIMATES risk"
    )


def interpretation(summary: pd.DataFrame) -> str:
    """One generated sentence per batch, built from the numbers."""
    items = [
        f"<li><b>{name}</b>: observed default {row.default_rate:.2%} vs predicted "
        f"{row.mean_risk_score:.2%} (gap {row.calibration_gap * 100:+.2f} p.p.): "
        f"{verdict(row.calibration_gap)}. ROC-AUC {row.roc_auc:.3f}.</li>"
        for name, row in summary.iterrows()
    ]
    return "<ul>" + "".join(items) + "</ul>"


def build_html(summary: pd.DataFrame, segments: dict, bands: dict) -> str:
    """Self-contained page: datasets, method, results, interpretation and limitations."""
    sections = [
        "<h1>Label-based performance monitoring</h1>",
        "<h2>Datasets and method</h2>",
        (
            "<p>Reference: baseline test set. Batches: control (no change), data_drift "
            "(inflation and unemployment) and concept_drift (same inputs, higher risk for "
            "clients with high payment burden). Calibration gap = observed default rate "
            f"- mean predicted risk; alert threshold: {config.CALIBRATION_GAP_ALERT:.0%}.</p>"
        ),
        "<h2>Interpretation</h2>",
        interpretation(summary),
        "<h2>Summary by batch</h2>",
        summary.round(4).to_html(),
        "<h2>Calibration by segment (diagnosis)</h2>",
    ]
    for name, table in segments.items():
        sections += [f"<h3>{name}</h3>", table.round(4).to_html()]
    sections.append("<h2>Calibration by score band (bands fixed on the reference)</h2>")
    for name, table in bands.items():
        sections += [f"<h3>{name}</h3>", table.round(4).to_html()]
    sections += [
        "<h2>Limitations</h2>",
        (
            "<ul><li>Labels are simulated; in production they arrive months after the "
            "decision.</li><li>Segments are known here because the scenario was designed; "
            "in practice they are investigated after a global alert.</li>"
            "<li>The unemployed default rate (29.4%) comes from only 17 reference clients.</li></ul>"
        ),
    ]
    style = (
        "<style>body{font-family:sans-serif;margin:2rem;max-width:960px}"
        "table{border-collapse:collapse;margin-bottom:1.5rem}"
        "td,th{border:1px solid #ccc;padding:4px 8px;text-align:right}</style>"
    )
    body = "".join(sections)
    return f"<!doctype html><html><head><meta charset='utf-8'>{style}</head><body>{body}</body></html>"


if __name__ == "__main__":
    import joblib

    from credit_monitoring.modeling.train import decide
    from credit_monitoring.monitoring.drift import build_reference

    baseline = json.loads(config.BASELINE_METRICS_PATH.read_text())
    threshold = baseline["decision_policy"]["threshold"]
    burden_limit = json.loads(config.SIMULATION_SUMMARY_PATH.read_text())[
        "concept_drift"
    ]["burden_limit"]

    # Reference: the baseline test set, scored and decided with the frozen threshold
    reference = build_reference(joblib.load(config.MODEL_PATH))
    reference["APPROVED"] = decide(reference["RISK_SCORE"].to_numpy(), threshold)

    batches = {"reference": reference} | {
        name: pd.read_csv(path) for name, path in BATCHES.items()
    }

    summary = pd.DataFrame({name: performance(df) for name, df in batches.items()}).T
    segments = {
        name: segment_calibration(df, burden_limit) for name, df in batches.items()
    }

    # Score bands from the reference deciles, then frozen for every batch
    inner = np.unique(np.quantile(reference["RISK_SCORE"], np.linspace(0, 1, 11))[1:-1])
    edges = np.concatenate([[-np.inf], inner, [np.inf]])
    bands = {name: calibration_by_band(df, edges) for name, df in batches.items()}

    output = {
        "summary": summary.to_dict(orient="index"),
        "segments": {
            name: table.to_dict(orient="index") for name, table in segments.items()
        },
    }
    config.PERFORMANCE_SUMMARY_PATH.write_text(
        json.dumps(output, indent=2, default=float)
    )
    config.PERFORMANCE_REPORT_PATH.write_text(
        build_html(summary, segments, bands), encoding="utf-8"
    )

    print(summary.round(4).to_string())
    for name, table in segments.items():
        print(f"\n{name}:\n{table.round(4).to_string()}")
    print(f"\nReport saved to {config.PERFORMANCE_REPORT_PATH}")
