"""Tests for the operational health snapshot and silent-degradation alert."""

from credit_monitoring.observability.dashboard import render_dashboard
from credit_monitoring.observability.health import (
    build_health,
    evaluate_batch,
    prometheus_text,
    summarize_pipeline_steps,
)


def _row(column: str, psi: float, level: str) -> dict:
    return {
        "column": column,
        "type": "numeric",
        "psi": psi,
        "psi_level": level,
        "ks_statistic": 0.0,
        "ks_pvalue": 1.0,
    }


def test_control_batch_has_no_alerts():
    result = evaluate_batch(
        "control",
        [_row("AMT_INCOME_TOTAL", 0.0003, "stable")],
        {
            "calibration_gap": 0.0001,
            "roc_auc": 0.73,
            "rows_scored": 100,
            "default_rate": 0.08,
            "mean_risk_score": 0.08,
            "default_rate_approved": 0.06,
        },
        {"rows": 100, "blocked_rows": 0, "alerts_by_rule": {}},
        baseline_auc=0.73,
    )
    assert result["alerts"] == []
    assert not result["flags"]["silent_degradation"]


def test_data_drift_raises_feature_alert_not_silent():
    result = evaluate_batch(
        "data_drift",
        [_row("NAME_INCOME_TYPE", 0.56, "significant")],
        {"calibration_gap": 0.014, "roc_auc": 0.71, "rows_scored": 100},
        {"rows": 100, "blocked_rows": 0, "alerts_by_rule": {}},
        baseline_auc=0.73,
    )
    assert "feature_drift" in result["alerts"]
    assert "calibration_gap" in result["alerts"]
    assert not result["flags"]["silent_degradation"]


def test_concept_drift_is_silent_degradation():
    result = evaluate_batch(
        "concept_drift",
        [
            _row("AMT_INCOME_TOTAL", 0.0003, "stable"),
            _row("RISK_SCORE", 0.0002, "stable"),
        ],
        {"calibration_gap": 0.016, "roc_auc": 0.69, "rows_scored": 100},
        {"rows": 100, "blocked_rows": 0, "alerts_by_rule": {}},
        baseline_auc=0.73,
    )
    assert result["flags"]["silent_degradation"]
    assert "silent_degradation" in result["alerts"]
    assert "feature_drift" not in result["alerts"]
    assert result["flags"]["performance_drop"]


def test_build_health_and_dashboard_include_concept_drift_alert():
    health = build_health(
        drift_summary={
            "control": [_row("AMT_INCOME_TOTAL", 0.0003, "stable")],
            "concept_drift": [_row("AMT_INCOME_TOTAL", 0.0003, "stable")],
        },
        performance_summary={
            "summary": {
                "control": {
                    "calibration_gap": 0.0001,
                    "roc_auc": 0.73,
                    "rows_scored": 10,
                },
                "concept_drift": {
                    "calibration_gap": 0.016,
                    "roc_auc": 0.69,
                    "rows_scored": 10,
                },
            }
        },
        production_summary={
            "control": {"rows": 10, "blocked_rows": 0, "alerts_by_rule": {}},
            "concept_drift": {"rows": 10, "blocked_rows": 0, "alerts_by_rule": {}},
        },
        log_records=[
            {
                "event": "pipeline_step_finished",
                "task": "predict",
                "batch": "control",
                "status": "success",
                "duration_seconds": 1.2,
                "rows": 10,
            }
        ],
        baseline={"roc_auc": 0.73},
    )
    assert "silent_degradation" in health["active_alerts"]
    html = render_dashboard(health)
    assert "concept_drift" in html
    assert "Degradação silenciosa" in html
    text = prometheus_text(health)
    assert 'credit_silent_degradation{batch="concept_drift"} 1' in text
    assert 'credit_pipeline_task_status{task="predict",batch="control"} 1' in text


def test_failed_pipeline_step_becomes_an_alert():
    steps = summarize_pipeline_steps(
        [
            {
                "event": "pipeline_step_finished",
                "task": "train",
                "status": "failure",
                "duration_seconds": 0.1,
            }
        ]
    )
    health = build_health({}, {}, log_records=[
        {
            "event": "pipeline_step_finished",
            "task": "train",
            "status": "failure",
            "duration_seconds": 0.1,
        }
    ])
    assert steps[0]["status"] == "failure"
    assert "pipeline_task_failed" in health["active_alerts"]
