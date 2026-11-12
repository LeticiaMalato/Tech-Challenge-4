"""Tests for the baseline model pipeline (synthetic data, no dataset needed)."""

import numpy as np
import pandas as pd

from credit_monitoring import config
from credit_monitoring.modeling.train import build_model, prepare_features


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
