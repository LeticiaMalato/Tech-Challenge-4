"""Structured pipeline logs: human-readable on the console, JSON Lines on disk.

The JSON file is the source for the observability dashboard and for any log
stack (Loki, CloudWatch, Datadog). Client identifiers must never be logged:
use the quarantine module, which HMAC-pseudonymizes IDs before writing.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Iterator

from credit_monitoring import config

LOGGER_NAME = "credit_monitoring"
_RESERVED = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "taskName",
    "asctime",
}


def safe_extra(payload: dict) -> dict:
    """Prefix keys that would collide with LogRecord attributes."""
    return {
        (f"ctx_{key}" if key in _RESERVED else key): value
        for key, value in payload.items()
    }


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with `extra` fields promoted to the top level."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key in payload or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def setup_logging(path=None) -> logging.Logger:
    """Idempotent: console + JSONL file. Creates the logs directory if needed."""
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(console)

    log_path = path or config.PIPELINE_LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)
    return logger


def get_logger() -> logging.Logger:
    return setup_logging()


@contextmanager
def pipeline_step(task: str, **fields) -> Iterator[dict]:
    """Time a pipeline task and emit start/finish events (success or failure)."""
    logger = get_logger()
    started = time.perf_counter()
    result: dict = {"status": "success"}
    logger.info(
        "pipeline step started",
        extra=safe_extra({"event": "pipeline_step_started", "task": task, **fields}),
    )
    try:
        yield result
    except Exception as exc:
        result["status"] = "failure"
        result["error"] = str(exc)
        logger.exception(
            "pipeline step failed",
            extra=safe_extra(
                {
                    "event": "pipeline_step_failed",
                    "task": task,
                    "status": "failure",
                    "error": str(exc),
                    **fields,
                }
            ),
        )
        raise
    finally:
        duration = time.perf_counter() - started
        logger.info(
            "pipeline step finished",
            extra=safe_extra(
                {
                    "event": "pipeline_step_finished",
                    "task": task,
                    "status": result.get("status", "success"),
                    "duration_seconds": round(duration, 4),
                    **fields,
                    **{
                        key: value
                        for key, value in result.items()
                        if key not in {"status"}
                    },
                }
            ),
        )


def read_log_records(path=None) -> list[dict]:
    """Load the JSONL log file; skip blank or malformed lines."""
    log_path = path or config.PIPELINE_LOG_PATH
    if not log_path.exists():
        return []
    records = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records
