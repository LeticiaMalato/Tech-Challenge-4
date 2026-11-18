"""Keep pipeline logs and MLflow files out of the repository during tests."""

import logging

import pytest

from credit_monitoring import config


@pytest.fixture(autouse=True)
def _isolate_observability_paths(tmp_path, monkeypatch):
    logging.getLogger("credit_monitoring").handlers.clear()
    monkeypatch.setattr(config, "PIPELINE_LOG_PATH", tmp_path / "pipeline.jsonl")
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(config, "QUARANTINE_DIR", tmp_path / "quarantine")
    monkeypatch.setattr(config, "MLFLOW_DB_PATH", tmp_path / "mlflow.db")
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
