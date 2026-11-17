"""Bias audit: decisions and calibration by gender and age group.

Gender is not a model feature (non-discrimination); it is kept only for this audit.
Age is a model feature, so it is audited too. Questions answered per group:
- do the groups have different real risk? (default rate)
- are they approved at different rates? (approval rate)
- are GOOD payers approved at the same rate? (approval rate among good payers)
- is the model's risk equally well calibrated? (calibration gap)
"""

import json

import joblib
import numpy as np
import pandas as pd

from credit_monitoring import config
from credit_monitoring.modeling.train import decide
from credit_monitoring.monitoring.drift import build_reference
from credit_monitoring.observability.logging import pipeline_step


def group_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = {}
    for group, part in df.groupby(group_col, observed=True):
        y = part[config.TARGET_COLUMN].to_numpy()
        approved = part["APPROVED"].to_numpy(dtype=bool)
        good = y == 0
        rows[group] = {
            "clients": len(part),
            "default_rate": y.mean(),
            "approval_rate": approved.mean(),
            "default_rate_approved": y[approved].mean() if approved.any() else np.nan,
            "approval_rate_good_payers": approved[good].mean()
            if good.any()
            else np.nan,
            "mean_risk_score": part["RISK_SCORE"].mean(),
            "calibration_gap": y.mean() - part["RISK_SCORE"].mean(),
            "interpretable": len(part) >= config.MIN_SEGMENT_SIZE,
        }
    return pd.DataFrame(rows).T


def _audit(step: dict) -> None:
    baseline = json.loads(config.BASELINE_METRICS_PATH.read_text())
    threshold = baseline["decision_policy"]["threshold"]

    # Reference test set: out-of-sample scores, decisions with the frozen threshold
    reference = build_reference(joblib.load(config.MODEL_PATH))
    reference["APPROVED"] = decide(reference["RISK_SCORE"].to_numpy(), threshold)
    reference["AGE_BAND"] = pd.cut(
        reference["AGE_YEARS"], bins=config.AGE_BANDS, labels=config.AGE_BAND_LABELS
    )

    audit = {
        "gender": group_metrics(reference, "CODE_GENDER"),
        "age_band": group_metrics(reference, "AGE_BAND"),
    }
    config.BIAS_AUDIT_PATH.write_text(
        json.dumps(
            {k: v.to_dict(orient="index") for k, v in audit.items()},
            indent=2,
            default=str,
        )
    )
    for name, table in audit.items():
        print(f"\n=== {name} ===")
        print(table.round(4).to_string())
    step["rows"] = int(len(reference))


if __name__ == "__main__":
    with pipeline_step("bias_audit") as step:
        _audit(step)
