"""Tests for MLflow tracking: success path and failure isolation."""

import pytest

from credit_monitoring.observability.tracking import (
    log_monitoring_run,
    log_training_run,
)


def test_tracking_failure_does_not_raise(monkeypatch):
    def boom():
        raise RuntimeError("mlflow down")

    monkeypatch.setattr(
        "credit_monitoring.observability.tracking._configure_mlflow", boom
    )
    assert log_training_run({"model": "LogisticRegression", "roc_auc": 0.73}) is None
    assert log_monitoring_run("control", {"metrics": {"psi_max": 0.0}}) is None


def test_monitoring_run_is_recorded(monkeypatch, tmp_path):
    mlflow = pytest.importorskip("mlflow")

    db = tmp_path / "mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{db.as_posix()}")
    run_id = log_monitoring_run(
        "concept_drift",
        {
            "params": {"batch": "concept_drift"},
            "metrics": {"psi_max": 0.0003, "calibration_gap": 0.016},
            "flags": {"silent_degradation": True, "drift_alert": False},
        },
    )
    assert run_id
    run = mlflow.get_run(run_id)
    assert run.data.tags["run_type"] == "monitoring"
    assert run.data.tags["batch"] == "concept_drift"
    assert run.data.metrics["silent_degradation"] == 1.0
    assert abs(run.data.metrics["calibration_gap"] - 0.016) < 1e-9
