"""Tests for the baseline model pipeline (synthetic data, no dataset needed)."""

import numpy as np
import pandas as pd

from credit_monitoring import config
from credit_monitoring.modeling.train import (
    FEATURES,
    build_model,
    compute_threshold,
    decide,
    evaluate_policy,
    prepare_features,
)


def synthetic_data(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Random data with the model columns and both target classes."""
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(size=n) for col in config.NUMERIC_FEATURES}
    data["EMPLOYMENT_GROUP"] = rng.choice(["0-1", "1-3", "NOT_EMPLOYED"], size=n)
    data["CAR_AGE_GROUP"] = rng.choice(["0-5", "NO_CAR"], size=n)
    data["NAME_INCOME_TYPE"] = rng.choice(["Working", "Pensioner"], size=n)
    data["NAME_EDUCATION_TYPE"] = rng.choice(["Higher education", "Secondary"], size=n)
    data["NAME_CONTRACT_TYPE"] = rng.choice(["Cash loans", "Revolving loans"], size=n)
    X = prepare_features(pd.DataFrame(data))
    y = pd.Series(rng.integers(0, 2, size=n))
    return X, y


def test_model_outputs_valid_probabilities():
    X, y = synthetic_data()
    model = build_model().fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_missing_numeric_values_do_not_break_prediction():
    X, y = synthetic_data()
    model = build_model().fit(X, y)
    X_new = X.head(3).copy()
    X_new["EXT_SOURCE_1"] = np.nan
    assert len(model.predict_proba(X_new)) == 3


def test_unseen_category_does_not_break_prediction():
    X, y = synthetic_data()
    model = build_model().fit(X, y)
    X_new = X.head(1).copy()
    X_new["NAME_INCOME_TYPE"] = "Businessman"  # never seen in training
    assert len(model.predict_proba(X_new)) == 1


def test_gender_is_not_a_model_feature():
    """LGPD non-discrimination decision, enforced automatically."""
    assert "CODE_GENDER" not in FEATURES


def test_policy_respects_approval_rate_and_separates_risk():
    rng = np.random.default_rng(0)
    proba = rng.uniform(size=10_000)
    # Synthetic signal: higher predicted risk → higher chance of default
    y_true = (rng.uniform(size=10_000) < proba * 0.3).astype(int)

    threshold = compute_threshold(proba, 0.9)
    policy = evaluate_policy(y_true, proba, threshold)

    assert abs(policy["approval_rate"] - 0.9) < 0.01
    assert policy["default_rate_approved"] < policy["default_rate_rejected"]


def test_decide_works_without_labels():
    approved = decide(np.array([0.05, 0.20, 0.50]), threshold=0.17)
    assert approved.tolist() == [True, False, False]


def test_policy_without_defaults_returns_none_instead_of_nan():
    proba = np.array([0.05, 0.10, 0.50])
    policy = evaluate_policy(np.array([0, 0, 0]), proba, threshold=0.17)
    assert policy["recall_risk"] is None


def test_policy_with_everyone_approved_returns_none_for_rejected():
    proba = np.array([0.05, 0.10])
    policy = evaluate_policy(np.array([0, 1]), proba, threshold=0.99)
    assert policy["default_rate_rejected"] is None
