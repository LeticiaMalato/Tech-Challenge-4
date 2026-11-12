"""Train and evaluate the baseline credit scoring model."""

import joblib
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
    blocked_rows = blocked.loc[blocked["index"] != BATCH, "index"].nunique()
    print(f"Reference rows: {len(raw)} | blocked by contract: {blocked_rows}")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")
    print(
        f"PR-AUC:  {average_precision_score(y_test, proba):.4f} "
        f"(random baseline = default rate = {y_test.mean():.4f})"
    )

    # 5. Save the trained pipeline
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")
