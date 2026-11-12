"""Train and evaluate the baseline credit scoring model."""

import json
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_monitoring import config
from credit_monitoring.data.contract import BATCH, is_batch_rejected, validate_batch
from credit_monitoring.data.features import build_features

FEATURES = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES


def build_model() -> Pipeline:
    """Preprocessing and model in one object: statistics are learned only in fit()."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("num", numeric, config.NUMERIC_FEATURES),
            # handle_unknown="ignore": a category never seen in training does not break predictions
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                config.CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select the model columns; categories as plain strings for the encoder."""
    X = df[FEATURES].copy()
    X[config.CATEGORICAL_FEATURES] = X[config.CATEGORICAL_FEATURES].astype(str)
    return X


if __name__ == "__main__":
    # 1. Same path as production data: raw → contract → features
    raw = pd.read_csv(config.REFERENCE_RAW_PATH)
    valid, blocked, _ = validate_batch(raw)
    if is_batch_rejected(blocked):
        raise SystemExit(
            "Reference batch rejected by the data contract:\n"
            + blocked.to_string(index=False)
        )
    data = build_features(valid)

    X = prepare_features(data)
    y = data[config.TARGET_COLUMN]

    # 2. Train/test split inside the reference (stratified, reproducible)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )

    # 3. Fit: imputation, scaling and encoding are learned on the training set only
    model = build_model()
    model.fit(X_train, y_train)

    # 4. Evaluate on the test set (threshold-free metrics)
    proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    blocked_rows = blocked.loc[blocked["index"] != BATCH, "index"].nunique()
    print(f"Reference rows: {len(raw)} | blocked by contract: {blocked_rows}")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(
        f"PR-AUC:  {pr_auc:.4f} (random baseline = default rate = {y_test.mean():.4f})"
    )

    # 5. Decision threshold: learned on the TRAINING set, then frozen
    train_proba = model.predict_proba(X_train)[:, 1]
    y_true = y_test.to_numpy()

    def evaluate_policy(rate: float) -> dict:
        """Approve clients whose risk is below the train quantile for this approval rate."""
        threshold = float(np.quantile(train_proba, rate))
        approved = proba < threshold
        rejected = ~approved
        return {
            "approval_rate_target": rate,
            "threshold": threshold,
            "approval_rate_test": float(approved.mean()),
            "default_rate_approved": float(y_true[approved].mean()),
            "default_rate_rejected": float(y_true[rejected].mean()),
            "precision_risk": float(
                y_true[rejected].mean()
            ),  # rejected who really default
            "recall_risk": float(
                y_true[rejected].sum() / y_true.sum()
            ),  # defaulters caught
        }

    comparison = pd.DataFrame(
        [evaluate_policy(r) for r in config.APPROVAL_RATES_TO_COMPARE]
    )
    print("\nApproval policy comparison (test set):")
    print(comparison.round(4).to_string(index=False))

    policy = evaluate_policy(config.APPROVAL_RATE)

    # 6. Save the model and the baseline metrics (the reference for monitoring)
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)

    metrics = {
        "trained_at": datetime.now(UTC).isoformat(),
        "model": "LogisticRegression",
        "random_state": config.RANDOM_STATE,
        "reference_rows": len(raw),
        "blocked_rows": int(blocked_rows),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "default_rate_test": float(y_test.mean()),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "decision_policy": policy,
        "features": FEATURES,
    }
    config.REPORTS_DIR.mkdir(exist_ok=True)
    config.BASELINE_METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nModel saved to {config.MODEL_PATH}")
    print(f"Baseline metrics saved to {config.BASELINE_METRICS_PATH}")
