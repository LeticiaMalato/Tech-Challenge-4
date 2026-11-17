"""Tests for the quarantine of rows blocked by the data contract."""

import pandas as pd

from credit_monitoring import config
from credit_monitoring.data.contract import BATCH, validate_batch
from credit_monitoring.observability.quarantine import write_quarantine
from tests.test_contract import valid_row


def test_quarantine_pseudonymizes_id_and_keeps_the_original_out(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "QUARANTINE_DIR", tmp_path)
    monkeypatch.setenv(config.PSEUDONYM_KEY_ENV, "test-secret-key")

    raw = valid_row(SK_ID_CURR=100002, DAYS_BIRTH=int(-16 * config.DAYS_PER_YEAR))
    _, blocked, _ = validate_batch(raw)
    path = write_quarantine(raw, blocked, "underage")

    saved = pd.read_csv(path)
    assert config.ID_COLUMN in saved.columns
    assert str(100002) not in saved[config.ID_COLUMN].astype(str).tolist()
    assert "100002" not in path.read_text(encoding="utf-8")


def test_empty_blocked_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "QUARANTINE_DIR", tmp_path)
    raw = valid_row()
    _, blocked, _ = validate_batch(raw)
    assert write_quarantine(raw, blocked, "ok") is None
    assert list(tmp_path.glob("*")) == []


def test_structural_rejection_writes_summary_without_row_payload(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "QUARANTINE_DIR", tmp_path)
    raw = valid_row()
    blocked = pd.DataFrame(
        [{"index": BATCH, "column": "AMT_INCOME_TOTAL", "check": "column_in_dataframe"}]
    )
    path = write_quarantine(raw, blocked, "broken")
    payload = path.read_text(encoding="utf-8")
    assert "structural" in payload
    assert "column_in_dataframe" in payload
    # The raw income of the client is not dumped into the structural summary
    assert "150000" not in payload
