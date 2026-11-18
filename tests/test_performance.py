"""Tests for label-based performance monitoring (synthetic data)."""

import numpy as np
import pandas as pd

from credit_monitoring.monitoring.performance import performance, verdict


def scored_batch(
    default_multiplier: float, n: int = 20_000, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    proba = rng.uniform(0.02, 0.20, size=n)
    target = (rng.uniform(size=n) < proba * default_multiplier).astype(int)
    return pd.DataFrame(
        {"RISK_SCORE": proba, "TARGET": target, "APPROVED": proba < 0.15}
    )


def test_calibrated_batch_has_near_zero_gap():
    result = performance(scored_batch(default_multiplier=1.0))
    assert abs(result["calibration_gap"]) < 0.01
    assert verdict(result["calibration_gap"]) == "calibrated"


def test_doubled_default_gives_positive_gap():
    result = performance(scored_batch(default_multiplier=2.0))
    assert result["calibration_gap"] > 0.05
    assert "UNDERESTIMATES" in verdict(result["calibration_gap"])
