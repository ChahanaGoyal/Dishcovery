"""
tests/test_memory_optimisation.py
==================================
Tests for the memory and caching optimisations introduced to fix the
recurring Streamlit Cloud OOM crash:

1. ``FOODCOM_LOAD_COLUMNS`` is defined and contains only the 5 expected columns.
2. ``load_csv`` accepts ``usecols`` and returns only those columns.
3. ``load_csv`` with ``usecols=FOODCOM_LOAD_COLUMNS`` omits the 7 high-memory
   columns that are unused by Dishcovery.
4. ``get_tfidf_index`` signature takes no arguments (stable cache key).
5. ``download_dataset_from_hf`` retries on failure and raises ``RuntimeError``
   after all retries are exhausted.
6. ``IngredientRecommender.fit()`` does not duplicate the DataFrame.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from src.data_loader import (
    FOODCOM_LOAD_COLUMNS,
    FOODCOM_REQUIRED_COLUMNS,
    download_dataset_from_hf,
    load_csv,
)
from src.recommender import IngredientRecommender


# ── Fixtures ──────────────────────────────────────────────────────────────────

_FULL_CSV = """\
name,id,minutes,contributor_id,submitted,tags,nutrition,n_steps,steps,description,ingredients,n_ingredients
Chocolate Cake,1,45,999,2000-01-01,['dessert'],['100'],3,['mix'],Nice cake.,"['flour', 'sugar', 'cocoa']",3
Tomato Soup,2,20,888,2000-02-01,['soup'],['80'],2,['chop'],Yum.,"['tomatoes', 'onion', 'broth']",3
"""


def _write_csv(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "RAW_recipes.csv"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ── 1. FOODCOM_LOAD_COLUMNS definition ───────────────────────────────────────


class TestFoodcomLoadColumns:
    def test_load_columns_is_list(self) -> None:
        assert isinstance(FOODCOM_LOAD_COLUMNS, list)

    def test_load_columns_count(self) -> None:
        """Only the 5 columns Dishcovery needs should be listed."""
        assert len(FOODCOM_LOAD_COLUMNS) == 5, (
            f"Expected 5 columns in FOODCOM_LOAD_COLUMNS, got {len(FOODCOM_LOAD_COLUMNS)}: "
            f"{FOODCOM_LOAD_COLUMNS}"
        )

    def test_required_columns_subset_of_load_columns(self) -> None:
        """Every required column must be in the load columns."""
        for col in FOODCOM_REQUIRED_COLUMNS:
            assert col in FOODCOM_LOAD_COLUMNS, (
                f"Required column '{col}' missing from FOODCOM_LOAD_COLUMNS"
            )

    def test_tags_in_load_columns(self) -> None:
        """'tags' must be loaded for dietary classification."""
        assert "tags" in FOODCOM_LOAD_COLUMNS

    def test_heavy_columns_excluded(self) -> None:
        """High-memory unused columns must NOT be in FOODCOM_LOAD_COLUMNS."""
        excluded = {"contributor_id", "submitted", "nutrition", "n_steps", "steps", "description", "n_ingredients"}
        overlap = excluded.intersection(set(FOODCOM_LOAD_COLUMNS))
        assert not overlap, (
            f"Heavy unused columns found in FOODCOM_LOAD_COLUMNS: {overlap} — "
            "these waste ~230 MB of RAM on Streamlit Cloud."
        )


# ── 2 & 3. load_csv usecols support ──────────────────────────────────────────


class TestLoadCsvUsecols:
    def test_usecols_restricts_columns(self, tmp_path: Path) -> None:
        """load_csv with usecols returns only the requested columns (order may vary)."""
        csv_file = _write_csv(tmp_path, _FULL_CSV)
        df = load_csv(csv_file, usecols=["id", "name", "ingredients"])
        assert set(df.columns) == {"id", "name", "ingredients"}

    def test_usecols_none_returns_all_columns(self, tmp_path: Path) -> None:
        """load_csv without usecols returns all columns (backward-compat)."""
        csv_file = _write_csv(tmp_path, _FULL_CSV)
        df = load_csv(csv_file)
        # The full CSV has 12 columns
        assert len(df.columns) == 12

    def test_foodcom_load_columns_omits_heavy_columns(self, tmp_path: Path) -> None:
        """Using FOODCOM_LOAD_COLUMNS drops all 7 unused high-memory columns."""
        csv_file = _write_csv(tmp_path, _FULL_CSV)
        df = load_csv(csv_file, usecols=FOODCOM_LOAD_COLUMNS)
        heavy = {"contributor_id", "submitted", "nutrition", "n_steps", "steps", "description", "n_ingredients"}
        assert not heavy.intersection(set(df.columns)), (
            f"Unexpected heavy columns loaded: {heavy.intersection(set(df.columns))}"
        )

    def test_foodcom_load_columns_includes_required_columns(self, tmp_path: Path) -> None:
        """Using FOODCOM_LOAD_COLUMNS still includes all FOODCOM_REQUIRED_COLUMNS."""
        csv_file = _write_csv(tmp_path, _FULL_CSV)
        df = load_csv(csv_file, usecols=FOODCOM_LOAD_COLUMNS)
        for col in FOODCOM_REQUIRED_COLUMNS:
            assert col in df.columns, f"Required column '{col}' missing after usecols filter"

    def test_usecols_row_count_unchanged(self, tmp_path: Path) -> None:
        """usecols does not affect row count."""
        csv_file = _write_csv(tmp_path, _FULL_CSV)
        df_full = load_csv(csv_file)
        df_slim = load_csv(csv_file, usecols=FOODCOM_LOAD_COLUMNS)
        assert len(df_full) == len(df_slim)


# ── 4. get_tfidf_index signature ─────────────────────────────────────────────


class TestGetTfidfIndexSignature:
    def test_get_tfidf_index_takes_no_args(self) -> None:
        """get_tfidf_index must accept zero arguments (stable @st.cache_resource key).

        Passing a large DataFrame as an argument caused @st.cache_resource to hash
        ~200 MB on every call, defeating caching and triggering repeated expensive
        refits on Streamlit Cloud.
        """
        import inspect
        import app as app_module

        sig = inspect.signature(app_module.get_tfidf_index)
        # The function may be wrapped by @st.cache_resource; check the underlying
        # function's parameters by inspecting the original source if available.
        params = [
            p for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
        ]
        assert len(params) == 0, (
            f"get_tfidf_index should require zero arguments (stable cache key). "
            f"Found required params: {[p.name for p in params]}. "
            "Passing a DataFrame argument defeats @st.cache_resource caching."
        )


# ── 5. download_dataset_from_hf retry logic ───────────────────────────────────


class TestDownloadDatasetFromHfRetry:
    def test_retries_on_failure_and_raises(self) -> None:
        """download_dataset_from_hf raises RuntimeError after all retries exhausted.

        hf_hub_download is imported lazily inside the function, so we patch it
        at the huggingface_hub module level.
        """
        import src.data_loader as dl_mod

        call_count = 0

        def _failing_download(**kwargs):
            nonlocal call_count
            call_count += 1
            raise ConnectionError("simulated network error")

        import huggingface_hub
        with patch.object(huggingface_hub, "hf_hub_download", side_effect=_failing_download):
            with pytest.raises(RuntimeError, match="Could not download"):
                dl_mod.download_dataset_from_hf(max_retries=1)
            # max_retries=1 → 2 total attempts
            assert call_count == 2, f"Expected 2 attempts (1 retry), got {call_count}"

    def test_succeeds_on_second_attempt(self, tmp_path: Path) -> None:
        """download_dataset_from_hf returns the path when a retry succeeds."""
        import src.data_loader as dl_mod

        attempt = 0
        fake_path = tmp_path / "RAW_recipes.csv"
        fake_path.write_text("id,name\n1,Cake\n")

        def _flaky_download(**kwargs):
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise ConnectionError("first attempt fails")
            return str(fake_path)

        import huggingface_hub
        with patch.object(huggingface_hub, "hf_hub_download", side_effect=_flaky_download):
            result = dl_mod.download_dataset_from_hf(max_retries=1)
            assert Path(result) == fake_path
            assert attempt == 2


# ── 6. IngredientRecommender.fit() does not copy the DataFrame ────────────────


class TestIngredientRecommenderFit:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "id": [1, 2],
            "name": ["Cake", "Soup"],
            "ingredients": ["['flour', 'sugar']", "['tomatoes', 'onion']"],
            "minutes": [45, 20],
        })

    def test_fit_does_not_duplicate_dataframe(self) -> None:
        """IngredientRecommender.fit() must not call df.copy().

        Calling df.copy() on a 200 MB catalog needlessly duplicates memory.
        The recommender stores a reference; the catalog is already owned by
        the @st.cache_data cache.
        """
        df = self._make_df()
        recommender = IngredientRecommender()
        recommender.fit(df)
        # The internal _recipes must be the same object, not a copy
        assert recommender._recipes is df, (
            "IngredientRecommender._recipes should be the same DataFrame object "
            "(not a copy) to avoid duplicating memory. "
            "Remove df.copy() from IngredientRecommender.fit()."
        )

    def test_fit_returns_self(self) -> None:
        df = self._make_df()
        recommender = IngredientRecommender()
        result = recommender.fit(df)
        assert result is recommender

    def test_fit_marks_recommender_as_fitted(self) -> None:
        df = self._make_df()
        recommender = IngredientRecommender()
        assert not recommender.is_fitted
        recommender.fit(df)
        assert recommender.is_fitted
