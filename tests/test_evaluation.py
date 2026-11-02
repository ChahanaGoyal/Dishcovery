"""
tests/test_evaluation.py
=========================
Unit tests for src.evaluation metrics and evaluation workflows.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import get_data_dir
from src.evaluation import (
    evaluate_recommender,
    f1_at_k,
    precision_at_k,
    recall_at_k,
    run_baseline_evaluation,
)

RECOMMENDED = ["r1", "r2", "r3", "r4", "r5"]
RELEVANT = {"r1", "r3", "r5"}


class TestPrecisionAtK:
    def test_perfect(self):
        assert precision_at_k(["r1", "r3", "r5"], {"r1", "r3", "r5"}, k=3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["r2", "r4"], {"r1", "r3"}, k=2) == 0.0

    def test_partial(self):
        # 2 out of 3 are relevant
        result = precision_at_k(["r1", "r2", "r3"], {"r1", "r3"}, k=3)
        assert abs(result - 2 / 3) < 1e-9

    def test_k_larger_than_list(self):
        # Should not crash; numerator still uses k
        result = precision_at_k(["r1"], {"r1"}, k=5)
        assert result == 1 / 5

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            precision_at_k(RECOMMENDED, RELEVANT, k=0)


class TestRecallAtK:
    def test_perfect(self):
        assert recall_at_k(["r1", "r3", "r5"], {"r1", "r3", "r5"}, k=3) == 1.0

    def test_empty_relevant(self):
        assert recall_at_k(RECOMMENDED, set(), k=3) == 0.0

    def test_partial(self):
        # 2 out of 3 relevant found in top-3
        result = recall_at_k(["r1", "r2", "r3"], {"r1", "r3", "r5"}, k=3)
        assert abs(result - 2 / 3) < 1e-9

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            recall_at_k(RECOMMENDED, RELEVANT, k=-1)


class TestF1AtK:
    def test_perfect(self):
        assert f1_at_k(["r1", "r3", "r5"], {"r1", "r3", "r5"}, k=3) == 1.0

    def test_zero_when_no_hits(self):
        assert f1_at_k(["r2", "r4"], {"r1", "r3"}, k=2) == 0.0

    def test_harmonic_mean(self):
        p = precision_at_k(RECOMMENDED, RELEVANT, k=3)
        r = recall_at_k(RECOMMENDED, RELEVANT, k=3)
        expected = 2 * p * r / (p + r)
        assert abs(f1_at_k(RECOMMENDED, RELEVANT, k=3) - expected) < 1e-9


# ── evaluate_recommender Tests ────────────────────────────────────────────────


@pytest.fixture
def mock_recipe_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "recipe_id": ["r1", "r2", "r3", "r4"],
            "title": ["Salad", "Soup", "Pizza", "Pasta"],
            "ingredients": [
                "lettuce, tomato, olive oil",
                "carrot, onion, broth",
                "flour, yeast, tomato, cheese",
                "pasta, garlic, olive oil",
            ],
            "minutes": [10, 30, 45, 20],
        }
    )


class TestEvaluateRecommender:
    def test_evaluates_mock_dataset(self, mock_recipe_df):
        """Mock dataset evaluation with partial inputs correctly reports all required metrics."""
        summary = evaluate_recommender(
            df=mock_recipe_df,
            sample_size=3,
            top_n=3,
            random_state=42,
            partial_ratio=0.5,
            verbose=False,
        )
        assert summary["valid_test_cases"] == 3
        assert summary["total_evaluated"] == 3
        assert summary["successful_retrievals"] + summary["unsuccessful_retrievals"] == 3
        assert 0.0 <= summary["retrieval_rate"] <= 100.0
        assert summary["average_ranking_position"] is not None
        assert 0.0 <= summary["average_match_percentage"] <= 100.0
        assert summary["partial_ratio"] == 0.5
        assert summary["top_n_cutoff"] == 3

    def test_evaluates_mock_dataset_full_input(self, mock_recipe_df):
        """Full ingredient input scenario (sanity check) yields 100% match."""
        summary = evaluate_recommender(
            df=mock_recipe_df,
            sample_size=3,
            top_n=3,
            random_state=42,
            partial_ratio=1.0,
            verbose=False,
        )
        assert summary["valid_test_cases"] == 3
        assert summary["retrieval_rate"] == 100.0
        assert summary["average_match_percentage"] == 100.0

    def test_invalid_partial_ratio_raises_value_error(self, mock_recipe_df):
        """Invalid partial_ratio raises clear ValueError."""
        with pytest.raises(ValueError, match="partial_ratio must be a number in"):
            evaluate_recommender(mock_recipe_df, partial_ratio=0.0)
        with pytest.raises(ValueError, match="partial_ratio must be a number in"):
            evaluate_recommender(mock_recipe_df, partial_ratio=-0.2)
        with pytest.raises(ValueError, match="partial_ratio must be a number in"):
            evaluate_recommender(mock_recipe_df, partial_ratio=1.5)

    def test_evaluates_empty_dataframe_safely(self):
        """Empty DataFrame does not divide by zero and returns clean zero metrics."""
        summary = evaluate_recommender(
            df=pd.DataFrame(),
            sample_size=10,
            verbose=False,
        )
        assert summary["total_evaluated"] == 0
        assert summary["retrieval_rate"] == 0.0
        assert summary["average_match_percentage"] == 0.0

    def test_evaluates_none_dataframe_safely(self):
        """None DataFrame returns clean zero metrics without crashing."""
        summary = evaluate_recommender(
            df=None,  # type: ignore
            sample_size=10,
            verbose=False,
        )
        assert summary["total_evaluated"] == 0
        assert summary["retrieval_rate"] == 0.0

    def test_evaluates_dataframe_without_usable_ingredients(self):
        """Dataframe with only empty/missing ingredients handled safely."""
        df = pd.DataFrame(
            {
                "recipe_id": ["1", "2"],
                "title": ["Empty 1", "Empty 2"],
                "ingredients": ["", None],
            }
        )
        summary = evaluate_recommender(df, sample_size=5, verbose=False)
        assert summary["total_evaluated"] == 0
        assert summary["retrieval_rate"] == 0.0

    def test_reproducible_sample_selection(self, mock_recipe_df):
        """Repeated runs with the same seed yield identical results."""
        run1 = evaluate_recommender(mock_recipe_df, sample_size=2, random_state=99, verbose=False)
        run2 = evaluate_recommender(mock_recipe_df, sample_size=2, random_state=99, verbose=False)
        assert run1 == run2

    def test_does_not_mutate_dataset(self, mock_recipe_df):
        """Input dataset is not modified by evaluate_recommender."""
        df_copy = mock_recipe_df.copy(deep=True)
        evaluate_recommender(mock_recipe_df, sample_size=3, verbose=False)
        pd.testing.assert_frame_equal(mock_recipe_df, df_copy)

    def test_real_dataset_run_baseline_evaluation(self):
        """Integration test with real dataset file."""
        raw_path = get_data_dir("raw") / "RAW_recipes.csv"
        if not raw_path.exists():
            pytest.skip("RAW_recipes.csv not found locally.")

        summary = run_baseline_evaluation(
            dataset_path=raw_path,
            nrows=200,
            sample_size=10,
            top_n=5,
            random_state=42,
        )
        assert summary["total_evaluated"] == 10
        assert summary["successful_retrievals"] >= 0
        assert 0.0 <= summary["retrieval_rate"] <= 100.0
