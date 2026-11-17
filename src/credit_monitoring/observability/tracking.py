"""MLflow tracking for training and monitoring runs.

SQLite is the local backend: MLflow 3 keeps the file store in maintenance mode
and the Model Registry needs a database. A tracking failure is logged and
swallowed — observability must not take the pipeline down.
"""

from __future__ import annotations

import os
from pathlib import Path

from credit_monitoring import config
from credit_monitoring.observability.logging import get_logger, safe_extra

# MLflow 3 serializes sklearn with skops; numpy.dtype appears in StandardScaler.
SKOPS_TRUSTED_TYPES = ["numpy.dtype"]


def tracking_uri() -> str:
    env = os.environ.get("MLFLOW_TRACKING_URI")
    if env:
        return env
    db = config.MLFLOW_DB_PATH.resolve().as_posix()
    return f"sqlite:///{db}"


def _configure_mlflow():
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    return mlflow


def _log_model(mlflow, model) -> None:
    """Log the sklearn Pipeline; fall back to the joblib file if skops refuses it."""
    try:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=config.MLFLOW_REGISTERED_MODEL,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        return
    except Exception as exc:
        get_logger().warning(
            "mlflow sklearn.log_model skipped",
            extra=safe_extra({"error": str(exc)}),
        )
    if config.MODEL_PATH.exists():
        mlflow.log_artifact(str(config.MODEL_PATH), artifact_path="model")


def log_training_run(metrics: dict, model=None, artifact_paths: list[Path] | None = None):
    """One run for the frozen baseline. Returns the run id, or None on failure."""
    logger = get_logger()
    try:
        mlflow = _configure_mlflow()
        with mlflow.start_run(run_name="training") as run:
            mlflow.set_tag("run_type", "training")
            mlflow.log_param("model", metrics.get("model", "LogisticRegression"))
            mlflow.log_param("random_state", metrics.get("random_state"))
            mlflow.log_param(
                "approval_rate_target",
                metrics.get("decision_policy", {}).get("approval_rate_target"),
            )
            mlflow.log_param(
                "threshold", metrics.get("decision_policy", {}).get("threshold")
            )
            for key in ("roc_auc", "pr_auc", "default_rate_test"):
                if key in metrics:
                    mlflow.log_metric(key, float(metrics[key]))
            policy = metrics.get("decision_policy") or {}
            for key in (
                "approval_rate",
                "default_rate_approved",
                "recall_risk",
            ):
                if policy.get(key) is not None:
                    mlflow.log_metric(f"policy_{key}", float(policy[key]))
            if model is not None:
                _log_model(mlflow, model)
            for path in artifact_paths or []:
                if Path(path).exists():
                    mlflow.log_artifact(str(path))
            return run.info.run_id
    except Exception as exc:
        logger.warning(
            "mlflow training run failed", extra=safe_extra({"error": str(exc)})
        )
        return None


def log_monitoring_run(batch: str, payload: dict, artifact_paths: list[Path] | None = None):
    """One run per production batch, compared against the frozen baseline."""
    logger = get_logger()
    try:
        mlflow = _configure_mlflow()
        with mlflow.start_run(run_name=f"monitoring-{batch}") as run:
            mlflow.set_tag("run_type", "monitoring")
            mlflow.set_tag("batch", batch)
            params = payload.get("params") or {}
            for key, value in params.items():
                mlflow.log_param(key, value)
            metrics = payload.get("metrics") or {}
            for key, value in metrics.items():
                if value is None:
                    continue
                try:
                    mlflow.log_metric(key, float(value))
                except (TypeError, ValueError):
                    continue
            flags = payload.get("flags") or {}
            for key, value in flags.items():
                mlflow.set_tag(key, str(bool(value)).lower())
                mlflow.log_metric(key, 1.0 if value else 0.0)
            for path in artifact_paths or []:
                if Path(path).exists():
                    mlflow.log_artifact(str(path))
            return run.info.run_id
    except Exception as exc:
        logger.warning(
            "mlflow monitoring run failed",
            extra=safe_extra({"batch": batch, "error": str(exc)}),
        )
        return None
