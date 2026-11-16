"""Tests for client ID pseudonymization."""

import pandas as pd
import pytest

from credit_monitoring.governance.privacy import get_pseudonym_key, pseudonymize

IDS = pd.Series([100002, 100003, 100002])


def test_same_id_and_key_give_same_pseudonym():
    result = pseudonymize(IDS, b"key-a")
    assert result.iloc[0] == result.iloc[2]
    assert result.iloc[0] != result.iloc[1]


def test_original_id_does_not_appear():
    result = pseudonymize(IDS, b"key-a")
    assert not result.str.contains("100002").any()


def test_different_keys_give_different_pseudonyms():
    assert (pseudonymize(IDS, b"key-a") != pseudonymize(IDS, b"key-b")).all()


def test_missing_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("CREDIT_MONITORING_PSEUDONYM_KEY", raising=False)
    with pytest.raises(RuntimeError, match="PSEUDONYM_KEY"):
        get_pseudonym_key()
