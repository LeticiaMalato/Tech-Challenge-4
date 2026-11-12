"""Score production batches: contract → features → model → frozen decision.

Two batches are scored with the same frozen model and threshold:
- control: production without changes (expected to behave like the reference)
- drifted: production after the simulated economic shift
"""

import json

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from credit_monitoring import config
from credit_monitoring.data.contract import BATCH, is_batch_rejected, validate_batch
from credit_monitoring.data.features import build_features
from credit_monitoring.modeling.train import decide, evaluate_policy, prepare_features


def score_batch(
    raw: pd.DataFrame, model, threshold: float
) -> tuple[pd.DataFrame, dict]:
    """Validate, transform, score and decide. Label metrics only if TARGET exists."""
    valid, blocked, alerts = validate_batch(raw)
    if is_batch_rejected(blocked):
        raise SystemExit(
            "Batch rejected by the data contract:\n" + blocked.to_string(index=False)
        )

    data = build_features(valid)
    proba = model.predict_proba(prepare_features(data))[:, 1]
    approved = decide(proba, threshold)

    scored = data.copy()
    scored["RISK_SCORE"] = proba
    scored["APPROVED"] = approved

    # What the fintech sees on the day (no labels needed)
    summary = {
        "rows": len(raw),
        "blocked_rows": int(blocked.loc[blocked["index"] != BATCH, "index"].nunique()),
        "alerts_by_rule": {
            rule: int(n)
            for rule, n in alerts.groupby("check")["index"].nunique().items()
        },
        "mean_risk_score": float(proba.mean()),
        "approval_rate": float(approved.mean()),
    }

    # What the fintech only learns months later (labels)
    if config.TARGET_COLUMN in data.columns:
        y_true = data[config.TARGET_COLUMN].to_numpy()
        summary["with_labels"] = {
            "default_rate": float(y_true.mean()),
            "roc_auc": float(roc_auc_score(y_true, proba)),
            "pr_auc": float(average_precision_score(y_true, proba)),
            **evaluate_policy(y_true, proba, threshold),
        }
    return scored, summary


if __name__ == "__main__":
    model = joblib.load(config.MODEL_PATH)
    baseline = json.loads(config.BASELINE_METRICS_PATH.read_text())
    threshold = baseline["decision_policy"]["threshold"]  # frozen: never recomputed

    batches = {
        "control": (config.PRODUCTION_RAW_PATH, config.PREDICTIONS_CONTROL_PATH),
        "drifted": (config.PRODUCTION_DRIFTED_PATH, config.PREDICTIONS_DRIFTED_PATH),
    }
    summaries = {"threshold": threshold}
    for name, (input_path, output_path) in batches.items():
        scored, summary = score_batch(pd.read_csv(input_path), model, threshold)
        scored.to_csv(output_path, index=False)
        summaries[name] = summary

    config.PRODUCTION_SUMMARY_PATH.write_text(json.dumps(summaries, indent=2))

    # Side-by-side comparison: reference (test set) x control x drifted
    rows = {
        "reference": {
            "approval_rate": baseline["decision_policy"]["approval_rate"],
            "default_rate": baseline["default_rate_test"],
            "default_rate_approved": baseline["decision_policy"][
                "default_rate_approved"
            ],
            "roc_auc": baseline["roc_auc"],
            "pr_auc": baseline["pr_auc"],
        }
    }
    for name in batches:
        s = summaries[name]
        rows[name] = {
            "approval_rate": s["approval_rate"],
            "default_rate": s["with_labels"]["default_rate"],
            "default_rate_approved": s["with_labels"]["default_rate_approved"],
            "roc_auc": s["with_labels"]["roc_auc"],
            "pr_auc": s["with_labels"]["pr_auc"],
        }
    print(f"Frozen threshold: {threshold:.4f}\n")
    print(pd.DataFrame(rows).T.round(4).to_string())
    print("\nAlerts by rule:")
    for name in batches:
        print(f"  {name}: {summaries[name]['alerts_by_rule']}")
