"""Consolidate logs, Etapa 2 reports and MLflow into the Etapa 3 deliverable."""

from __future__ import annotations

import json

from credit_monitoring import config
from credit_monitoring.observability.dashboard import render_dashboard
from credit_monitoring.observability.health import health_from_reports, prometheus_text
from credit_monitoring.observability.logging import get_logger, pipeline_step, safe_extra
from credit_monitoring.observability.tracking import log_monitoring_run


def main() -> None:
    logger = get_logger()  # also creates logs/pipeline.jsonl
    with pipeline_step("observability") as step:
        health = health_from_reports()
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        config.HEALTH_SUMMARY_PATH.write_text(
            json.dumps(health, indent=2, default=str), encoding="utf-8"
        )
        config.OBSERVABILITY_DASHBOARD_PATH.write_text(
            render_dashboard(health), encoding="utf-8"
        )
        config.PROMETHEUS_METRICS_PATH.write_text(
            prometheus_text(health), encoding="utf-8"
        )

        run_ids = {}
        for batch in health["batches"]:
            payload = {
                "params": {"batch": batch["batch"]},
                "metrics": {
                    "psi_max": batch["psi_max"],
                    "calibration_gap": batch["calibration_gap"],
                    "blocked_rows": batch["blocked_rows"],
                    "rows": batch["rows"],
                    "roc_auc": batch.get("roc_auc"),
                    "auc_delta": batch.get("auc_delta"),
                    "default_rate_approved": batch.get("default_rate_approved"),
                    "drifted_feature_count": batch["drifted_feature_count"],
                },
                "flags": batch["flags"],
            }
            artifacts = [
                config.DRIFT_SUMMARY_PATH,
                config.REPORTS_DIR / f"drift_report_{batch['batch']}.html",
                config.PERFORMANCE_SUMMARY_PATH,
                config.HEALTH_SUMMARY_PATH,
            ]
            run_ids[batch["batch"]] = log_monitoring_run(
                batch["batch"], payload, artifact_paths=artifacts
            )

        step["rows"] = sum(batch["rows"] for batch in health["batches"])
        step["alert_count"] = health["alert_count"]
        logger.info(
            "observability snapshot written",
            extra=safe_extra(
                {
                    "event": "observability_snapshot",
                    "dashboard": str(config.OBSERVABILITY_DASHBOARD_PATH),
                    "health": str(config.HEALTH_SUMMARY_PATH),
                    "mlflow_runs": run_ids,
                    "active_alerts": health["active_alerts"],
                }
            ),
        )
        print(f"Dashboard: {config.OBSERVABILITY_DASHBOARD_PATH}")
        print(f"Health:    {config.HEALTH_SUMMARY_PATH}")
        print(f"Metrics:   {config.PROMETHEUS_METRICS_PATH}")
        print("Active alerts:", ", ".join(health["active_alerts"]) or "(none)")
        for batch, run_id in run_ids.items():
            print(f"  MLflow {batch}: {run_id or 'not recorded'}")


if __name__ == "__main__":
    main()
