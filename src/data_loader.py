"""
data_loader.py
==============
Utilities for loading and validating raw recipe data.

This module is intentionally thin at project initialisation.
Dataset ingestion logic will be added once a dataset is confirmed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Expected columns in a valid recipe dataset
REQUIRED_COLUMNS: list[str] = ["recipe_id", "title", "ingredients"]


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV recipe file and return a validated DataFrame.

    Parameters
    ----------
    path:
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least the columns defined in ``REQUIRED_COLUMNS``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the CSV is missing required columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return df


def get_data_dir(subdir: str = "raw") -> Path:
    """Return the absolute path to a data sub-directory.

    Parameters
    ----------
    subdir:
        Either ``"raw"`` or ``"processed"``.
    """
    root = Path(__file__).resolve().parents[1]  # project root
    return root / "data" / subdir
