"""Operational health snapshot: pipeline logs + drift + performance → alerts.

This is the single place that decides what is an alert. The dashboard, the
Prometheus textfile and the MLflow monitoring runs all read the same verdict.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from credit_monitoring import config
from credit_monitoring.observability.logging import read_log_records


def _max_psi(drift_rows: list[dict]) -> float:
    return max((float(row["psi"]) for row in drift_rows), default=0.0)


def _drifted_features(drift_rows: list[dict]) -> list[str]:
    return [
        row["column"]
        for row in drift_rows
        if row.get("psi_level") in {"moderate", "significant"}
    ]


def evaluate_batch(
    name: str,
    drift_rows: list[dict],
    performance: dict | None,
    production: dict | None,
    baseline_auc: float | None = None,
) -> dict:
    """Alerts for one scored batch. Silent degradation needs BOTH monitors."""
    psi_max = _max_psi(drift_rows)
    drifted = _drifted_features(drift_rows)
    gap = float((performance or {}).get("calibration_gap") or 0.0)
    auc = (performance or {}).get("roc_auc")
    blocked = int((production or {}).get("blocked_rows") or 0)
    rows = int(
        (production or {}).get("rows")
        or (performance or {}).get("rows_scored")
        or 0
    )
    alerts_by_rule = (production or {}).get("alerts_by_rule") or {}
    contract_alerts = int(sum(alerts_by_rule.values()))

    drift_alert = psi_max >= config.PSI_MODERATE
    calibration_alert = abs(gap) >= config.CALIBRATION_GAP_ALERT
    silent_degradation = (not drift_alert) and calibration_alert
    auc_drop = None
    performance_drop = False
    if baseline_auc is not None and auc is not None:
        auc_drop = float(auc) - float(baseline_auc)
        performance_drop = auc_drop <= -config.ROC_AUC_DROP_ALERT

    alerts = []
    if blocked:
        alerts.append("contract_rows_blocked")
    if drift_alert:
        alerts.append("feature_drift")
    if calibration_alert:
        alerts.append("calibration_gap")
    if silent_degradation:
        alerts.append("silent_degradation")
    if performance_drop:
        alerts.append("performance_drop")

    psi_by_feature = {row["column"]: float(row["psi"]) for row in drift_rows}

    return {
        "batch": name,
        "rows": rows,
        "blocked_rows": blocked,
        "contract_alerts": contract_alerts,
        "alerts_by_rule": alerts_by_rule,
        "psi_max": psi_max,
        "psi_by_feature": psi_by_feature,
        "drifted_features": drifted,
        "drifted_feature_count": len(drifted),
        "calibration_gap": gap,
        "roc_auc": auc,
        "auc_delta": auc_drop,
        "default_rate": (performance or {}).get("default_rate"),
        "mean_risk_score": (performance or {}).get("mean_risk_score"),
        "default_rate_approved": (performance or {}).get("default_rate_approved"),
        "flags": {
            "drift_alert": drift_alert,
            "calibration_alert": calibration_alert,
            "silent_degradation": silent_degradation,
            "performance_drop": performance_drop,
            "contract_blocked": blocked > 0,
        },
        "alerts": alerts,
    }


def summarize_pipeline_steps(records: list[dict]) -> list[dict]:
    """Keep the finish event of each task (last occurrence wins)."""
    steps: dict[str, dict] = {}
    for record in records:
        if record.get("event") != "pipeline_step_finished":
            continue
        task = record.get("task") or record.get("ctx_name") or "unknown"
        key = f"{task}:{record.get('batch', '')}"
        steps[key] = {
            "task": task,
            "batch": record.get("batch"),
            "status": record.get("status", "success"),
            "duration_seconds": record.get("duration_seconds"),
            "timestamp": record.get("timestamp"),
            "rows": record.get("rows"),
            "error": record.get("error"),
        }
    return list(steps.values())


def build_health(
    drift_summary: dict,
    performance_summary: dict,
    production_summary: dict | None = None,
    log_records: list[dict] | None = None,
    baseline: dict | None = None,
) -> dict:
    """Aggregate reports + logs into one operational snapshot."""
    performance = (performance_summary or {}).get("summary") or {}
    production = production_summary or {}
    baseline_auc = (baseline or {}).get("roc_auc")
    batches = []
    for name, drift_rows in (drift_summary or {}).items():
        batches.append(
            evaluate_batch(
                name,
                drift_rows or [],
                performance.get(name),
                production.get(name) if isinstance(production.get(name), dict) else None,
                baseline_auc=baseline_auc,
            )
        )

    steps = summarize_pipeline_steps(log_records or [])
    failed_steps = [step for step in steps if step.get("status") != "success"]
    active_alerts = sorted({alert for batch in batches for alert in batch["alerts"]})
    if failed_steps:
        active_alerts.append("pipeline_task_failed")

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "metrics_documented": documented_metrics(),
        "pipeline": {
            "steps": steps,
            "failed_steps": len(failed_steps),
            "last_step_timestamp": steps[-1]["timestamp"] if steps else None,
        },
        "batches": batches,
        "active_alerts": active_alerts,
        "alert_count": len(active_alerts),
    }


def documented_metrics() -> list[dict]:
    """The operational metrics this project actually watches (Etapa 3)."""
    return [
        {
            "name": "pipeline_task_status",
            "source": "logs",
            "question": "Did this pipeline task finish successfully?",
        },
        {
            "name": "pipeline_task_duration_seconds",
            "source": "logs",
            "question": "How long did ingestion, scoring or drift detection take?",
        },
        {
            "name": "rows_processed",
            "source": "logs / production_summary",
            "question": "What volume did the batch carry?",
        },
        {
            "name": "blocked_rows",
            "source": "data contract",
            "question": "How many rows were quarantined before scoring?",
        },
        {
            "name": "contract_alerts",
            "source": "data contract",
            "question": "How many suspicious-but-accepted rows need review?",
        },
        {
            "name": "psi_max / feature PSI",
            "source": "Etapa 2 drift tests",
            "question": "Did the input distribution move (available on the decision day)?",
        },
        {
            "name": "calibration_gap",
            "source": "Etapa 2 performance (labels)",
            "question": "Is the model underestimating risk once labels arrive?",
        },
        {
            "name": "silent_degradation",
            "source": "PSI stable AND calibration_gap ≥ 1 p.p.",
            "question": "Is the model failing while input-drift monitors stay green?",
        },
        {
            "name": "roc_auc delta vs baseline",
            "source": "performance vs baseline_metrics",
            "question": "Did ranking quality drop beyond 0.03 AUC?",
        },
    ]


def prometheus_text(health: dict) -> str:
    """OpenMetrics text the Prometheus node exporter can scrape from a file."""
    lines = [
        "# HELP credit_pipeline_task_status 1 = success, 0 = failure",
        "# TYPE credit_pipeline_task_status gauge",
    ]
    for step in health["pipeline"]["steps"]:
        task = _label(step["task"])
        batch = _label(step.get("batch") or "none")
        status = 1 if step.get("status") == "success" else 0
        lines.append(
            f'credit_pipeline_task_status{{task="{task}",batch="{batch}"}} {status}'
        )
        if step.get("duration_seconds") is not None:
            lines.append(
                "# HELP credit_pipeline_task_duration_seconds Task wall time"
            )
            lines.append(
                "# TYPE credit_pipeline_task_duration_seconds gauge"
            )
            lines.append(
                "credit_pipeline_task_duration_seconds"
                f'{{task="{task}",batch="{batch}"}} {step["duration_seconds"]}'
            )

    lines += [
        "# HELP credit_feature_psi PSI per feature (from the last health snapshot)",
        "# TYPE credit_feature_psi gauge",
        "# HELP credit_max_psi Worst PSI in the batch",
        "# TYPE credit_max_psi gauge",
        "# HELP credit_calibration_gap Observed default minus mean predicted risk",
        "# TYPE credit_calibration_gap gauge",
        "# HELP credit_silent_degradation 1 when inputs look stable but labels show miscalibration",
        "# TYPE credit_silent_degradation gauge",
        "# HELP credit_drift_alert 1 when any feature PSI is moderate or significant",
        "# TYPE credit_drift_alert gauge",
        "# HELP credit_blocked_rows Rows quarantined by the data contract",
        "# TYPE credit_blocked_rows gauge",
        "# HELP credit_rows_processed Batch size before scoring",
        "# TYPE credit_rows_processed gauge",
    ]
    # Feature-level PSI is not stored in evaluate_batch; only max. That is enough
    # for alerting. The HTML dashboard shows the full Etapa 2 table via reports.
    for batch in health["batches"]:
        name = _label(batch["batch"])
        lines.append(f'credit_max_psi{{batch="{name}"}} {batch["psi_max"]}')
        lines.append(
            f'credit_calibration_gap{{batch="{name}"}} {batch["calibration_gap"]}'
        )
        silent = 1 if batch["flags"]["silent_degradation"] else 0
        drift = 1 if batch["flags"]["drift_alert"] else 0
        lines.append(f'credit_silent_degradation{{batch="{name}"}} {silent}')
        lines.append(f'credit_drift_alert{{batch="{name}"}} {drift}')
        lines.append(f'credit_blocked_rows{{batch="{name}"}} {batch["blocked_rows"]}')
        lines.append(f'credit_rows_processed{{batch="{name}"}} {batch["rows"]}')
        if batch.get("roc_auc") is not None:
            lines.append(
                "# HELP credit_model_roc_auc Ranking quality on the batch"
            )
            lines.append("# TYPE credit_model_roc_auc gauge")
            lines.append(
                f'credit_model_roc_auc{{batch="{name}"}} {batch["roc_auc"]}'
            )
    lines.append("# EOF")
    # Deduplicate HELP/TYPE that we repeated in the loop
    seen_help: set[str] = set()
    cleaned = []
    for line in lines:
        if line.startswith("# HELP ") or line.startswith("# TYPE "):
            if line in seen_help:
                continue
            seen_help.add(line)
        cleaned.append(line)
    return "\n".join(cleaned) + "\n"


def _label(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def load_json(path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def health_from_reports() -> dict:
    """Build the snapshot from the artifacts already produced by Etapas 1–2."""
    return build_health(
        drift_summary=load_json(config.DRIFT_SUMMARY_PATH),
        performance_summary=load_json(config.PERFORMANCE_SUMMARY_PATH),
        production_summary=load_json(config.PRODUCTION_SUMMARY_PATH),
        log_records=read_log_records(),
        baseline=load_json(config.BASELINE_METRICS_PATH),
    )
