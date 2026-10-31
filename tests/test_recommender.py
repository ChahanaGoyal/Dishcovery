"""
tests/test_recommender.py
=========================
Unit and integration tests for src.recommender.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import get_data_dir, load_csv
from src.recommender import (
    IngredientRecommender,
    Recipe,
    calculate_match_score,
    match_recipes,
    rank_recipes,
)


# ── Sample Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def sample_recipes_df() -> pd.DataFrame:
    """Sample DataFrame with known ingredient overlaps."""
    return pd.DataFrame(
        {
            "recipe_id": ["r1", "r2", "r3", "r4"],
            "title": [
                "Tomato Salad",
                "Classic Tomato Soup",
                "Spaghetti Bolognese",
                "Fruit Bowl",
            ],
            "ingredients": [
                "tomato, olive oil, salt",                      # 3 ings
                "tomato, onion, broth, cream",                  # 4 ings
                "ground beef, onion, tomato, garlic, pasta",    # 5 ings
                "apple, banana, orange",                        # 3 ings
            ],
            "minutes": [10, 30, 45, 5],
        }
    )


# ── Task 1: calculate_match_score Tests ────────────────────────────────────────


class TestCalculateMatchScore:
    def test_all_recipe_ingredients_available(self):
        """Case 1: All recipe ingredients are available (100% match)."""
        pct, matched, missing = calculate_match_score(
            user_ingredients=["tomato", "salt"],
            recipe_ingredients=["tomato", "salt"],
        )
        assert pct == 100.0
        assert set(matched) == {"tomato", "salt"}
        assert missing == []

    def test_some_recipe_ingredients_available(self):
        """Case 2: Some recipe ingredients are available (example from prompt: 3 of 4 = 75%)."""
        pct, matched, missing = calculate_match_score(
            user_ingredients=["tomato", "onion", "salt"],
            recipe_ingredients=["tomato", "onion", "salt", "potato"],
        )
        assert pct == 75.0
        assert set(matched) == {"tomato", "onion", "salt"}
        assert missing == ["potato"]

    def test_no_recipe_ingredients_match(self):
        """Case 3: No recipe ingredients match (0% match)."""
        pct, matched, missing = calculate_match_score(
            user_ingredients=["apple", "banana"],
            recipe_ingredients=["chicken", "rice"],
        )
        assert pct == 0.0
        assert matched == []
        assert set(missing) == {"chicken", "rice"}

    def test_empty_user_ingredients(self):
        """Case 4: User enters empty ingredient list or None."""
        pct1, matched1, missing1 = calculate_match_score([], ["tomato", "basil"])
        assert pct1 == 0.0
        assert matched1 == []
        assert set(missing1) == {"tomato", "basil"}

        pct2, matched2, missing2 = calculate_match_score(None, ["tomato", "basil"])
        assert pct2 == 0.0
        assert matched2 == []
        assert set(missing2) == {"tomato", "basil"}

    def test_empty_or_missing_recipe_ingredients(self):
        """Case 5: Recipe has empty or missing ingredients (division by zero avoided)."""
        pct1, matched1, missing1 = calculate_match_score(["tomato"], [])
        assert pct1 == 0.0
        assert matched1 == []
        assert missing1 == []

        pct2, matched2, missing2 = calculate_match_score(["tomato"], "")
        assert pct2 == 0.0
        assert matched2 == []
        assert missing2 == []

        pct3, matched3, missing3 = calculate_match_score(["tomato"], None)
        assert pct3 == 0.0
        assert matched3 == []
        assert missing3 == []

    def test_case_and_whitespace_insensitivity(self):
        """Case 6: Ingredient names differ only in capitalization or whitespace."""
        pct, matched, missing = calculate_match_score(
            user_ingredients=["  TOMATO  ", "Garlic"],
            recipe_ingredients=["tomato", "  garlic  "],
        )
        assert pct == 100.0
        assert set(matched) == {"tomato", "garlic"}
        assert missing == []

    def test_duplicate_recipe_ingredients_counted_once(self):
        """Duplicate recipe ingredients are deduplicated and not counted multiple times."""
        pct, matched, missing = calculate_match_score(
            user_ingredients=["tomato"],
            recipe_ingredients=["tomato", "Tomato", "onion"],
        )
        # Unique ingredients: tomato, onion (2 total). 1 matched = 50%.
        assert pct == 50.0
        assert matched == ["tomato"]
        assert missing == ["onion"]

    def test_stringified_list_recipe_ingredients(self):
        """Supports stringified lists like Food.com RAW_recipes.csv."""
        pct, matched, missing = calculate_match_score(
            user_ingredients=["butter", "honey"],
            recipe_ingredients="['winter squash', 'honey', 'butter', 'salt']",
        )
        # 2 out of 4 = 50.0%
        assert pct == 50.0
        assert set(matched) == {"butter", "honey"}
        assert set(missing) == {"winter squash", "salt"}

    def test_scores_remain_bounded_between_zero_and_hundred(self):
        """Scores always stay between 0.0 and 100.0."""
        inputs = [
            (["a"], ["a", "b", "c"]),
            (["a", "b", "c", "d"], ["a"]),
            ([], ["a"]),
            (["a"], []),
        ]
        for user_in, recipe_in in inputs:
            pct, _, _ = calculate_match_score(user_in, recipe_in)
            assert 0.0 <= pct <= 100.0


# ── Task 2: rank_recipes and Missing Ingredients Tests ────────────────────────


class TestRankRecipes:
    def test_higher_match_percentage_appears_first(self, sample_recipes_df):
        """Recipes with higher match percentages appear first."""
        # User has: tomato, onion, broth (matches 2/3 of r1 = 66.7%, 3/4 of r2 = 75%, 2/5 of r3 = 40%, 0/3 of r4 = 0%)
        results = rank_recipes(
            user_ingredients=["tomato", "onion", "broth"],
            df=sample_recipes_df,
            top_n=5,
        )
        assert len(results) == 4
        percentages = [r.match_percentage for r in results]
        assert percentages == sorted(percentages, reverse=True)
        assert results[0].recipe_id == "r2"  # 75%
        assert results[0].match_percentage == 75.0

    def test_missing_ingredients_calculated_accurately(self, sample_recipes_df):
        """Missing ingredients are correctly identified for each ranked recipe."""
        results = rank_recipes(
            user_ingredients=["tomato", "olive oil"],
            df=sample_recipes_df,
            top_n=1,
        )
        # r1 requires: tomato, olive oil, salt. User has tomato, olive oil. Missing: salt.
        top_recipe = results[0]
        assert top_recipe.recipe_id == "r1"
        assert top_recipe.missing_ingredients == ["salt"]
        assert set(top_recipe.matched_ingredients) == {"tomato", "olive oil"}

    def test_result_limit_top_n(self, sample_recipes_df):
        """Configurable result limit top_n limits returned items."""
        results = rank_recipes(
            user_ingredients=["tomato", "onion"],
            df=sample_recipes_df,
            top_n=2,
        )
        assert len(results) == 2

    def test_empty_user_input_returns_empty_list(self, sample_recipes_df):
        """Empty user input returns empty list gracefully."""
        assert rank_recipes([], sample_recipes_df) == []
        assert rank_recipes(["   ", ""], sample_recipes_df) == []
        assert rank_recipes(None, sample_recipes_df) == []

    def test_invalid_or_empty_recipe_data_does_not_crash(self):
        """Malformed rows (NaNs, empty ingredients, missing times) are handled safely."""
        dirty_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["A", "B", None],
                "ingredients": ["", None, "['salt']"],
                "minutes": [None, "invalid", 15],
            }
        )
        results = rank_recipes(["salt"], dirty_df, top_n=5)
        assert len(results) == 3
        # Recipe with salt should be top
        assert results[0].recipe_id == "3"
        assert results[0].match_percentage == 100.0
        assert results[0].cooking_time == 15

    def test_does_not_mutate_input_dataframe(self, sample_recipes_df):
        """Source DataFrame remains completely unmodified."""
        df_copy = sample_recipes_df.copy(deep=True)
        rank_recipes(["tomato", "onion"], sample_recipes_df)
        pd.testing.assert_frame_equal(sample_recipes_df, df_copy)

    def test_cooking_time_included_in_recipe_result(self, sample_recipes_df):
        """Cooking time is populated from minutes column."""
        results = rank_recipes(["apple", "banana", "orange"], sample_recipes_df, top_n=1)
        assert results[0].recipe_id == "r4"
        assert results[0].cooking_time == 5

    def test_match_recipes_helper(self, sample_recipes_df):
        """match_recipes returns match results for all recipes."""
        results = match_recipes(["tomato"], sample_recipes_df)
        assert len(results) == len(sample_recipes_df)

    def test_recipe_dataclass_methods(self):
        """Recipe dataclass properties and conversion methods."""
        rec = Recipe(
            recipe_id="101",
            title="Pasta",
            matched_ingredients=["pasta", "tomato"],
            match_percentage=66.67,
            missing_ingredients=["garlic"],
            cooking_time=20,
        )
        assert rec.name == "Pasta"
        assert rec.score == 0.6667
        d = rec.to_dict()
        assert d["recipe_id"] == "101"
        assert d["match_percentage"] == 66.67
        assert d["cooking_time"] == 20


# ── IngredientRecommender Class Tests ──────────────────────────────────────────


class TestIngredientRecommender:
    def test_fit_and_recommend(self, sample_recipes_df):
        """Standard recommender fit and recommend flow."""
        recommender = IngredientRecommender()
        assert not recommender.is_fitted
        recommender.fit(sample_recipes_df)
        assert recommender.is_fitted

        results = recommender.recommend(["tomato", "olive oil", "salt"], top_n=3)
        assert len(results) <= 3
        assert results[0].recipe_id == "r1"
        assert results[0].match_percentage == 100.0

    def test_unfitted_returns_empty(self):
        """Calling recommend on unfitted recommender returns empty list."""
        recommender = IngredientRecommender()
        assert recommender.recommend(["tomato"]) == []

    def test_fit_invalid_columns_raises_value_error(self):
        """fit raises ValueError when required columns are absent."""
        recommender = IngredientRecommender()
        bad_df = pd.DataFrame({"random_col": [1, 2]})
        with pytest.raises(ValueError, match="DataFrame must contain"):
            recommender.fit(bad_df)


# ── Integration Test with Real Dataset ─────────────────────────────────────────


class TestRealDatasetIntegration:
    def test_real_dataset_ranking_and_missing_ingredients(self):
        """Integration test with a sample from data/raw/RAW_recipes.csv."""
        raw_path = get_data_dir("raw") / "RAW_recipes.csv"
        if not raw_path.exists():
            pytest.skip("RAW_recipes.csv not found locally.")

        df = load_csv(
            raw_path,
            nrows=50,
            usecols=["id", "name", "minutes", "ingredients"],
        )
        assert len(df) == 50

        user_ingredients = ["butter", "eggs", "flour", "sugar", "milk"]
        results = rank_recipes(user_ingredients, df, top_n=5)

        assert len(results) <= 5
        assert len(results) > 0

        top = results[0]
        assert top.recipe_id != ""
        assert top.title != ""
        assert isinstance(top.match_percentage, float)
        assert 0.0 <= top.match_percentage <= 100.0
        assert isinstance(top.matched_ingredients, list)
        assert isinstance(top.missing_ingredients, list)
        if top.cooking_time is not None:
            assert isinstance(top.cooking_time, int)
