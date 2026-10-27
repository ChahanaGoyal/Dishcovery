"""
tests/test_data_loader.py
==========================
Unit tests for src.data_loader.

All tests use in-memory fixtures (tmp_path or StringIO) so they run without
the real Food.com dataset being present.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
    FOODCOM_REQUIRED_COLUMNS,
    get_data_dir,
    load_csv,
    validate_foodcom_columns,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_csv(path: Path, content: str) -> Path:
    """Write *content* to *path* and return *path*."""
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


# Minimal CSV that mirrors the Food.com RAW_recipes.csv structure.
_FOODCOM_CSV = """\
    id,name,minutes,ingredients,steps,tags
    1234,Chocolate Cake,45,"['flour', 'sugar', 'cocoa']","['mix', 'bake']","['dessert']"
    5678,Tomato Soup,20,"['tomatoes', 'onion', 'broth']","['chop', 'simmer']","['soup']"
"""


# ── load_csv ──────────────────────────────────────────────────────────────────


class TestLoadCsv:
    def test_returns_dataframe(self, tmp_path: Path) -> None:
        """load_csv returns a pd.DataFrame for a valid CSV."""
        csv_file = _write_csv(tmp_path / "recipes.csv", _FOODCOM_CSV)
        df = load_csv(csv_file)
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self, tmp_path: Path) -> None:
        """All data rows are loaded by default."""
        csv_file = _write_csv(tmp_path / "recipes.csv", _FOODCOM_CSV)
        df = load_csv(csv_file)
        assert len(df) == 2

    def test_column_names_present(self, tmp_path: Path) -> None:
        """Expected column names are present in the returned DataFrame."""
        csv_file = _write_csv(tmp_path / "recipes.csv", _FOODCOM_CSV)
        df = load_csv(csv_file)
        for col in ("id", "name", "minutes", "ingredients"):
            assert col in df.columns, f"Column '{col}' missing from DataFrame"

    def test_nrows_limits_rows(self, tmp_path: Path) -> None:
        """nrows parameter limits the number of rows loaded."""
        csv_file = _write_csv(tmp_path / "recipes.csv", _FOODCOM_CSV)
        df = load_csv(csv_file, nrows=1)
        assert len(df) == 1

    def test_usecols_limits_columns(self, tmp_path: Path) -> None:
        """usecols parameter loads only the requested columns."""
        csv_file = _write_csv(tmp_path / "recipes.csv", _FOODCOM_CSV)
        df = load_csv(csv_file, usecols=["id", "name"])
        assert list(df.columns) == ["id", "name"]

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """FileNotFoundError is raised for a non-existent path."""
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            load_csv(tmp_path / "does_not_exist.csv")

    def test_invalid_csv_raises_value_error(self, tmp_path: Path) -> None:
        """ValueError is raised when the file cannot be parsed as a valid CSV.

        We request a column (``usecols``) that does not exist in the file to
        trigger a pandas ``ValueError`` — a reliable cross-platform way to
        exercise the error-handling path without depending on how pandas
        handles binary content (which it reads silently).
        """
        csv_file = _write_csv(tmp_path / "recipes.csv", _FOODCOM_CSV)
        with pytest.raises(ValueError, match="Could not parse CSV file"):
            load_csv(csv_file, usecols=["nonexistent_column"])

    def test_sample_values(self, tmp_path: Path) -> None:
        """Spot-check a cell value to confirm data was read correctly."""
        csv_file = _write_csv(tmp_path / "recipes.csv", _FOODCOM_CSV)
        df = load_csv(csv_file)
        assert df.loc[0, "name"] == "Chocolate Cake"
        assert df.loc[0, "minutes"] == 45


# ── validate_foodcom_columns ──────────────────────────────────────────────────


class TestValidateFoodcomColumns:
    def test_passes_for_valid_dataframe(self) -> None:
        """No exception is raised when all required columns are present."""
        df = pd.DataFrame(columns=["id", "name", "minutes", "ingredients", "extra"])
        validate_foodcom_columns(df)  # should not raise

    def test_raises_for_missing_column(self) -> None:
        """ValueError is raised when a required column is absent."""
        df = pd.DataFrame(columns=["id", "name"])  # missing 'minutes', 'ingredients'
        with pytest.raises(ValueError, match="missing required Food.com columns"):
            validate_foodcom_columns(df)

    def test_error_lists_missing_columns(self) -> None:
        """The error message names the missing columns explicitly."""
        df = pd.DataFrame(columns=["id", "name"])
        with pytest.raises(ValueError) as exc_info:
            validate_foodcom_columns(df)
        msg = str(exc_info.value)
        assert "minutes" in msg or "ingredients" in msg

    def test_required_columns_constant(self) -> None:
        """FOODCOM_REQUIRED_COLUMNS contains the four essential fields."""
        for col in ("id", "name", "ingredients", "minutes"):
            assert col in FOODCOM_REQUIRED_COLUMNS


# ── get_data_dir ──────────────────────────────────────────────────────────────


class TestGetDataDir:
    def test_returns_path_object(self) -> None:
        """get_data_dir returns a pathlib.Path."""
        assert isinstance(get_data_dir(), Path)

    def test_default_is_raw(self) -> None:
        """Default subdirectory is 'raw'."""
        p = get_data_dir()
        assert p.parts[-1] == "raw"

    def test_processed_subdir(self) -> None:
        """'processed' subdirectory is returned when requested."""
        p = get_data_dir("processed")
        assert p.parts[-1] == "processed"

    def test_path_ends_with_data_subdir(self) -> None:
        """Returned path ends with data/raw."""
        p = get_data_dir("raw")
        assert p.parts[-2] == "data"
        assert p.parts[-1] == "raw"
