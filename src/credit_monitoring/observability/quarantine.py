"""Quarantine for rows blocked by the data contract.

The file lives under data/ (not versioned). Client IDs are HMAC-pseudonymized
before writing: the original SK_ID_CURR must never appear in logs or quarantine.
A structural batch rejection is recorded as a summary only — the whole batch is
untrusted, so dumping every row would copy personal data without a clear benefit.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from credit_monitoring import config
from credit_monitoring.data.contract import BATCH, is_batch_rejected
from credit_monitoring.governance.privacy import get_pseudonym_key, pseudonymize
from credit_monitoring.observability.logging import get_logger, safe_extra


def write_quarantine(
    raw: pd.DataFrame, blocked: pd.DataFrame, batch_name: str
) -> str | None:
    """Persist blocked rows (pseudonymized) or a structural-failure summary."""
    if blocked.empty:
        return None

    config.QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    logger = get_logger()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    if is_batch_rejected(blocked):
        path = config.QUARANTINE_DIR / f"{batch_name}_{stamp}_batch_rejected.json"
        summary = {
            "batch": batch_name,
            "rejected_at": datetime.now(UTC).isoformat(),
            "reason": "structural",
            "rows_in_batch": int(len(raw)),
            "failures": blocked.to_dict(orient="records"),
        }
        path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        logger.warning(
            "batch rejected by data contract",
            extra=safe_extra(
                {
                    "event": "batch_rejected",
                    "task": "ingest",
                    "batch": batch_name,
                    "rows": int(len(raw)),
                    "quarantine_path": str(path),
                    "checks": sorted(blocked["check"].astype(str).unique().tolist()),
                }
            ),
        )
        return str(path)

    indexes = blocked.loc[blocked["index"] != BATCH, "index"].unique()
    quarantined = raw.loc[raw.index.intersection(indexes)].copy()
    if quarantined.empty:
        return None

    key = get_pseudonym_key()
    if config.ID_COLUMN in quarantined.columns:
        quarantined[config.ID_COLUMN] = pseudonymize(quarantined[config.ID_COLUMN], key)

    path = config.QUARANTINE_DIR / f"{batch_name}_{stamp}.csv"
    quarantined.to_csv(path, index=False)
    logger.warning(
        "rows quarantined by data contract",
        extra=safe_extra(
            {
                "event": "rows_quarantined",
                "task": "ingest",
                "batch": batch_name,
                "blocked_rows": int(len(indexes)),
                "quarantine_path": str(path),
                "checks": sorted(blocked["check"].astype(str).unique().tolist()),
            }
        ),
    )
    return str(path)
