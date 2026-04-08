"""Utilities for loading and validating the email dataset."""
from __future__ import annotations
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent


def load_emails(path: str | Path | None = None) -> list[dict]:
    """
    Load the email dataset from JSON.

    Parameters
    ----------
    path : optional override path; defaults to data/emails.json
    """
    path = Path(path) if path else DATA_DIR / "emails.json"
    with path.open("r", encoding="utf-8") as f:
        emails = json.load(f)
    _validate(emails)
    return emails


def _validate(emails: list[dict]) -> None:
    required_keys = {"id", "sender", "subject", "body", "ground_truth"}
    for i, email in enumerate(emails):
        missing = required_keys - email.keys()
        if missing:
            raise ValueError(f"Email at index {i} missing keys: {missing}")
        gt = email["ground_truth"]
        if gt.get("label") not in ("spam", "urgent", "normal"):
            raise ValueError(
                f"Email {email['id']}: ground_truth.label must be spam/urgent/normal"
            )
