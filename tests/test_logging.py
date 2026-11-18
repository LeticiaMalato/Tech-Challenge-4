"""Tests for structured JSON pipeline logs."""

import json
import logging

from credit_monitoring.observability.logging import (
    JsonFormatter,
    pipeline_step,
    read_log_records,
    safe_extra,
    setup_logging,
)


def test_reserved_log_keys_are_prefixed():
    extra = safe_extra({"name": "drift", "batch": "control"})
    assert extra["ctx_name"] == "drift"
    assert extra["batch"] == "control"


def test_json_formatter_promotes_extra_fields():
    record = logging.LogRecord(
        name="credit_monitoring",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="pipeline step finished",
        args=(),
        exc_info=None,
    )
    record.task = "split"
    record.rows = 10
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "pipeline step finished"
    assert payload["task"] == "split"
    assert payload["rows"] == 10
    assert "timestamp" in payload


def test_pipeline_step_writes_jsonl(tmp_path):
    log_path = tmp_path / "pipeline.jsonl"
    logger = logging.getLogger("credit_monitoring")
    logger.handlers.clear()
    setup_logging(log_path)

    with pipeline_step("split") as step:
        step["rows"] = 42

    records = read_log_records(log_path)
    finished = [row for row in records if row.get("event") == "pipeline_step_finished"]
    assert len(finished) == 1
    assert finished[0]["task"] == "split"
    assert finished[0]["status"] == "success"
    assert finished[0]["rows"] == 42
    assert finished[0]["duration_seconds"] >= 0


def test_pipeline_step_failure_is_logged_and_reraised(tmp_path):
    log_path = tmp_path / "pipeline.jsonl"
    logging.getLogger("credit_monitoring").handlers.clear()
    setup_logging(log_path)

    try:
        with pipeline_step("train"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    else:
        raise AssertionError("pipeline_step must re-raise")

    records = read_log_records(log_path)
    failed = [row for row in records if row.get("event") == "pipeline_step_failed"]
    finished = [row for row in records if row.get("event") == "pipeline_step_finished"]
    assert failed and failed[0]["status"] == "failure"
    assert finished and finished[0]["status"] == "failure"
