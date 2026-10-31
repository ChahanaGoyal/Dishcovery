"""
data_loader.py
==============
Utilities for loading and validating raw recipe data.

The primary entry point is :func:`load_csv`, which reads any CSV file into a
Pandas DataFrame.  :func:`validate_foodcom_columns` then checks that the
Food.com ``RAW_recipes.csv`` columns required by Dishcovery are present.

Food.com column → Dishcovery name mapping
------------------------------------------
``id``          → recipe identifier
``name``        → recipe title
``ingredients`` → ingredient list (stored as a Python-list string in the CSV)
``minutes``     → total cooking time in minutes
``steps``       → ordered preparation steps
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that Dishcovery needs from RAW_recipes.csv.
# These are the *raw* Food.com column names (before any renaming).
FOODCOM_REQUIRED_COLUMNS: list[str] = [
    "id",
    "name",
    "ingredients",
    "minutes",
]


def load_csv(
    path: str | Path,
    nrows: int | None = None,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """Load a CSV file and return a Pandas DataFrame.

    Parameters
    ----------
    path:
        Absolute or relative path to the CSV file.
    nrows:
        If given, read only this many rows.  Useful for quickly inspecting a
        large file without loading it entirely into memory.
    usecols:
        If given, load only these columns.  Reduces memory usage when only a
        subset of columns is needed.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the requested rows and columns.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file cannot be parsed as a valid CSV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    try:
        df = pd.read_csv(path, nrows=nrows, usecols=usecols)
    except Exception as exc:
        raise ValueError(f"Could not parse CSV file '{path}': {exc}") from exc

    logger.info(
        "Loaded %d rows × %d columns from '%s'",
        len(df),
        len(df.columns),
        path,
    )
    return df


def validate_foodcom_columns(df: pd.DataFrame) -> None:
    """Check that *df* contains the Food.com columns Dishcovery requires.

    Parameters
    ----------
    df:
        DataFrame loaded from ``RAW_recipes.csv``.

    Raises
    ------
    ValueError
        If one or more required columns are absent.
    """
    missing = [col for col in FOODCOM_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required Food.com columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    logger.info("All required Food.com columns present: %s", FOODCOM_REQUIRED_COLUMNS)


def get_data_dir(subdir: str = "raw") -> Path:
    """Return the absolute path to a data sub-directory.

    Parameters
    ----------
    subdir:
        Either ``"raw"`` or ``"processed"``.
    """
    root = Path(__file__).resolve().parents[1]  # project root
    return root / "data" / subdir
