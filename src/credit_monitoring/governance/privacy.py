"""Pseudonymization of client identifiers for logs and quarantine files.

Client IDs are sequential integers with few possible values, so a plain hash is
reversible by brute force (hashing every possible ID). A keyed hash (HMAC-SHA256)
with a secret key prevents this. The result is still pseudonymized personal data
under the LGPD: whoever holds the key can re-link it.
"""

import hashlib
import hmac
import os

import pandas as pd

from credit_monitoring import config


def get_pseudonym_key() -> bytes:
    """Read the secret key from the environment. There is no default on purpose."""
    key = os.environ.get(config.PSEUDONYM_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"Set the {config.PSEUDONYM_KEY_ENV} environment variable "
            "before writing logs with client identifiers."
        )
    return key.encode()


def pseudonymize(ids: pd.Series, key: bytes) -> pd.Series:
    """Keyed hash of each ID; the same ID and key always give the same pseudonym."""
    return ids.astype(str).map(
        lambda value: hmac.new(key, value.encode(), hashlib.sha256).hexdigest()[:16]
    )
