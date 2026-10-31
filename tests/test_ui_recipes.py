"""
tests/test_ui_recipes.py
========================
Tests for Streamlit UI recipe loading, cooking-time filtering,
and ingredient formatting functions in app.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app import (
    filter_by_cooking_time,
    format_recipe_ingredients,
    parse_ingredients,
)


@pytest.fixture
def sample_recipe_df() -> pd.DataFrame:
    """Fixture providing a sample recipe DataFrame with diverse cooking times."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "name": [
                "Quick Salad",
                "Speedy Pasta",
                "Medium Curry",
                "Slow Roast",
                "Overnight Stew",
                "Negative Time Recipe",
                "Invalid Time Recipe",
                "Missing Time Recipe",
            ],
            "minutes": [10, 15, 30, 60, 120, -5, "not_a_number", np.nan],
            "ingredients": [
                "['lettuce', 'tomato', 'cucumber']",
                "['pasta', 'olive oil', 'garlic']",
                "['chicken', 'curry powder', 'onion']",
                "['beef', 'potatoes', 'carrots']",
                "['lamb', 'beans', 'broth']",
                "['mystery']",
                "['unknown']",
                "['salt']",
            ],
        }
    )


class TestFilterByCookingTime:
    def test_any_cooking_time_returns_all(self, sample_recipe_df: pd.DataFrame):
        result = filter_by_cooking_time(sample_recipe_df, "Any cooking time")
        assert len(result) == len(sample_recipe_df)
        assert list(result["id"]) == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_up_to_15_minutes(self, sample_recipe_df: pd.DataFrame):
        result = filter_by_cooking_time(sample_recipe_df, "Up to 15 minutes")
        # Should include 10 and 15, exclude >15, -5, string, and NaN
        assert len(result) == 2
        assert set(result["id"]) == {1, 2}

    def test_up_to_30_minutes(self, sample_recipe_df: pd.DataFrame):
        result = filter_by_cooking_time(sample_recipe_df, "Up to 30 minutes")
        # Should include 10, 15, 30
        assert len(result) == 3
        assert set(result["id"]) == {1, 2, 3}

    def test_up_to_60_minutes(self, sample_recipe_df: pd.DataFrame):
        result = filter_by_cooking_time(sample_recipe_df, "Up to 60 minutes")
        # Should include 10, 15, 30, 60
        assert len(result) == 4
        assert set(result["id"]) == {1, 2, 3, 4}

    def test_excludes_invalid_and_negative_times(self, sample_recipe_df: pd.DataFrame):
        for limit in ["Up to 15 minutes", "Up to 30 minutes", "Up to 60 minutes"]:
            result = filter_by_cooking_time(sample_recipe_df, limit)
            # Ensure negative, string, and nan IDs are never included
            assert 6 not in result["id"].values
            assert 7 not in result["id"].values
            assert 8 not in result["id"].values

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame(columns=["id", "name", "minutes", "ingredients"])
        result = filter_by_cooking_time(empty_df, "Up to 15 minutes")
        assert result.empty

    def test_no_matches_returns_empty_dataframe(self):
        long_df = pd.DataFrame({"id": [1], "name": ["Slow"], "minutes": [120]})
        result = filter_by_cooking_time(long_df, "Up to 15 minutes")
        assert result.empty

    def test_does_not_mutate_original_dataframe(self, sample_recipe_df: pd.DataFrame):
        original_copy = sample_recipe_df.copy()
        filter_by_cooking_time(sample_recipe_df, "Up to 15 minutes")
        pd.testing.assert_frame_equal(sample_recipe_df, original_copy)


class TestFormatRecipeIngredients:
    def test_stringified_list(self):
        raw = "['tomato', 'garlic', 'basil']"
        formatted = format_recipe_ingredients(raw)
        assert formatted == "tomato, garlic, basil"

    def test_python_list(self):
        raw = ["salt", "pepper", "olive oil"]
        formatted = format_recipe_ingredients(raw)
        assert formatted == "salt, pepper, olive oil"

    def test_plain_string(self):
        raw = "tomato, cheese"
        formatted = format_recipe_ingredients(raw)
        assert formatted == "tomato, cheese"

    def test_nan_and_none(self):
        assert format_recipe_ingredients(np.nan) == "Ingredients not available"
        assert format_recipe_ingredients(None) == "Ingredients not available"

    def test_empty_list(self):
        assert format_recipe_ingredients([]) == "Ingredients not available"
        assert format_recipe_ingredients("[]") == "Ingredients not available"


class TestIngredientInputValidation:
    def test_comma_separated(self):
        assert parse_ingredients("tomato, onion, potato") == ["tomato", "onion", "potato"]

    def test_strips_spaces(self):
        assert parse_ingredients(" tomato , onion ") == ["tomato", "onion"]

    def test_ignores_empty_items(self):
        assert parse_ingredients("tomato,,onion,") == ["tomato", "onion"]

    def test_deduplicates_preserving_order(self):
        assert parse_ingredients("tomato, tomato, onion") == ["tomato", "onion"]

    def test_empty_input(self):
        assert parse_ingredients("") == []

    def test_whitespace_and_commas_only(self):
        assert parse_ingredients("  ,  ,  ") == []
