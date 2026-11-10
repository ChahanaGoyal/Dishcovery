"""
tests/test_dietary.py
=====================
Tests for dietary classification and filtering:
  - classify_dietary() in src/preprocessing.py
  - filter_by_dietary() in src/preprocessing.py

Covers:
  - Any preference (all recipes returned).
  - Vegetarian filtering.
  - Non-vegetarian filtering.
  - Recipes with missing ingredient data.
  - Recipes with uncertain classification.
  - Compatibility with both recommendation methods (rank_recipes / rank_recipes_tfidf).
  - Compatibility with cooking-time filtering.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing import classify_dietary, filter_by_dietary


# ─────────────────────────────────────────────────────────────────────────────
# classify_dietary — unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyDietary:
    """Unit tests for src.preprocessing.classify_dietary."""

    # ── Vegetarian cases ──────────────────────────────────────────────────────

    def test_vegetarian_tag_returns_vegetarian(self):
        tags = ["vegetarian", "easy", "60-minutes-or-less"]
        ingredients = ["spinach", "garlic", "olive oil", "pasta"]
        assert classify_dietary(tags, ingredients) == "vegetarian"

    def test_vegetarian_tag_as_stringified_list(self):
        tags = "['vegetarian', 'healthy', 'low-fat']"
        ingredients = "['tomato', 'onion', 'cheese']"
        assert classify_dietary(tags, ingredients) == "vegetarian"

    def test_vegetarian_no_nonveg_ingredients(self):
        tags = ["vegetarian", "dairy-free"]
        ingredients = ["lentils", "tomato", "cumin", "coriander"]
        assert classify_dietary(tags, ingredients) == "vegetarian"

    # ── Non-vegetarian cases via tags ─────────────────────────────────────────

    def test_chicken_tag_returns_nonveg(self):
        tags = ["chicken", "30-minutes-or-less"]
        ingredients = ["chicken breast", "garlic", "olive oil"]
        assert classify_dietary(tags, ingredients) == "non-vegetarian"

    def test_beef_tag_returns_nonveg(self):
        tags = ["beef", "main-dish"]
        ingredients = ["ground beef", "onion", "tomato sauce"]
        assert classify_dietary(tags, ingredients) == "non-vegetarian"

    def test_seafood_tag_returns_nonveg(self):
        tags = ["seafood", "healthy"]
        ingredients = ["salmon", "lemon", "dill"]
        assert classify_dietary(tags, ingredients) == "non-vegetarian"

    def test_meat_tag_returns_nonveg(self):
        tags = ["meat", "easy"]
        ingredients = ["pork loin", "rosemary", "olive oil"]
        assert classify_dietary(tags, ingredients) == "non-vegetarian"

    # ── Non-vegetarian cases via ingredients ──────────────────────────────────

    def test_nonveg_ingredient_chicken_no_tag(self):
        tags = ["easy", "30-minutes-or-less"]
        ingredients = ["chicken thighs", "garlic", "lemon"]
        assert classify_dietary(tags, ingredients) == "non-vegetarian"

    def test_nonveg_ingredient_bacon_no_tag(self):
        tags = ["breakfast"]
        ingredients = ["bacon", "eggs", "cheese"]
        assert classify_dietary(tags, ingredients) == "non-vegetarian"

    def test_nonveg_ingredient_sausage_no_tag(self):
        tags = ["dinner"]
        ingredients = ["sausage", "peppers", "onion"]
        assert classify_dietary(tags, ingredients) == "non-vegetarian"

    def test_nonveg_ingredient_shrimp_no_tag(self):
        tags = ["asian"]
        ingredients = ["shrimp", "noodles", "soy sauce"]
        assert classify_dietary(tags, ingredients) == "non-vegetarian"

    def test_nonveg_ingredient_lard(self):
        tags = ["baking"]
        ingredients = ["flour", "lard", "sugar"]
        assert classify_dietary(tags, ingredients) == "non-vegetarian"

    def test_nonveg_ingredient_gelatin(self):
        tags = ["dessert"]
        ingredients = ["gelatin", "sugar", "fruit juice"]
        assert classify_dietary(tags, ingredients) == "non-vegetarian"

    # ── Vegetarian tag + non-veg ingredient → non-vegetarian wins ────────────

    def test_vegetarian_tag_overridden_by_nonveg_ingredient(self):
        """Non-veg ingredient keyword overrides a vegetarian tag."""
        tags = ["vegetarian"]
        ingredients = ["chicken", "tomato", "basil"]  # misclassified dataset row
        assert classify_dietary(tags, ingredients) == "non-vegetarian"

    # ── Uncertain cases ────────────────────────────────────────────────────────

    def test_no_vegetarian_tag_no_nonveg_ingredient_returns_uncertain(self):
        tags = ["easy", "quick"]
        ingredients = ["flour", "sugar", "butter", "eggs"]
        assert classify_dietary(tags, ingredients) == "uncertain"

    def test_empty_tags_empty_ingredients_returns_uncertain(self):
        assert classify_dietary([], []) == "uncertain"

    def test_none_tags_none_ingredients_returns_uncertain(self):
        assert classify_dietary(None, None) == "uncertain"

    def test_empty_string_tags_returns_uncertain(self):
        assert classify_dietary("", []) == "uncertain"

    # ── Missing data cases ────────────────────────────────────────────────────

    def test_none_tags_nonveg_ingredients(self):
        """Missing tags do not prevent non-veg detection from ingredients."""
        assert classify_dietary(None, ["beef", "onion"]) == "non-vegetarian"

    def test_none_ingredients_nonveg_tag(self):
        """Missing ingredients do not prevent non-veg detection from tags."""
        assert classify_dietary(["beef", "easy"], None) == "non-vegetarian"

    def test_none_ingredients_vegetarian_tag(self):
        """With vegetarian tag and no ingredients to check, returns vegetarian."""
        assert classify_dietary(["vegetarian", "easy"], None) == "vegetarian"

    def test_malformed_stringified_list(self):
        """Malformed stringified list is handled without crashing."""
        result = classify_dietary("not a list", "also not a list")
        # Cannot determine anything useful; should not crash
        assert result in {"vegetarian", "non-vegetarian", "uncertain"}

    def test_ingredients_empty_list(self):
        tags = ["vegetarian"]
        result = classify_dietary(tags, [])
        assert result == "vegetarian"

    # ── Stringified-list parsing ───────────────────────────────────────────────

    def test_stringified_ingredients_list(self):
        tags = ["chicken", "easy"]
        ingredients = "['chicken breast', 'garlic', 'olive oil']"
        assert classify_dietary(tags, ingredients) == "non-vegetarian"

    def test_stringified_vegetarian_ingredients(self):
        tags = "['vegetarian']"
        ingredients = "['spinach', 'garlic', 'pasta']"
        assert classify_dietary(tags, ingredients) == "vegetarian"


# ─────────────────────────────────────────────────────────────────────────────
# filter_by_dietary — unit tests (src.preprocessing.filter_by_dietary)
# ─────────────────────────────────────────────────────────────────────────────

def _make_recipe_df() -> pd.DataFrame:
    """Create a small test DataFrame with known dietary categories."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": [
                "Veggie Pasta",
                "Chicken Stir Fry",
                "Lentil Soup",
                "Beef Burger",
                "Plain Rice",
            ],
            "tags": [
                "['vegetarian', 'easy']",
                "['chicken', 'quick']",
                "['vegetarian', 'healthy']",
                "['beef', 'grilled']",
                "['quick', 'easy']",   # uncertain — no veg or nonveg signal
            ],
            "ingredients": [
                "['pasta', 'tomato', 'basil']",
                "['chicken breast', 'soy sauce', 'ginger']",
                "['lentils', 'onion', 'cumin']",
                "['ground beef', 'bun', 'lettuce']",
                "['rice', 'water', 'salt']",   # uncertain — no meat keyword
            ],
            "minutes": [30, 20, 45, 25, 10],
        }
    )


class TestFilterByDietary:
    """Unit tests for src.preprocessing.filter_by_dietary."""

    def test_any_returns_all_rows(self):
        df = _make_recipe_df()
        result = filter_by_dietary(df, "Any")
        assert len(result) == len(df)

    def test_vegetarian_returns_only_vegetarian(self):
        df = _make_recipe_df()
        result = filter_by_dietary(df, "Vegetarian")
        # Rows 0 (Veggie Pasta) and 2 (Lentil Soup) should be returned
        assert len(result) == 2
        assert set(result["name"]) == {"Veggie Pasta", "Lentil Soup"}

    def test_nonveg_returns_only_nonveg(self):
        df = _make_recipe_df()
        result = filter_by_dietary(df, "Non-vegetarian")
        # Rows 1 (Chicken) and 3 (Beef) should be returned
        assert len(result) == 2
        assert set(result["name"]) == {"Chicken Stir Fry", "Beef Burger"}

    def test_uncertain_excluded_from_vegetarian(self):
        """Plain Rice (uncertain) must not appear in Vegetarian results."""
        df = _make_recipe_df()
        result = filter_by_dietary(df, "Vegetarian")
        assert "Plain Rice" not in result["name"].values

    def test_uncertain_excluded_from_nonveg(self):
        """Plain Rice (uncertain) must not appear in Non-vegetarian results."""
        df = _make_recipe_df()
        result = filter_by_dietary(df, "Non-vegetarian")
        assert "Plain Rice" not in result["name"].values

    def test_empty_dataframe_any(self):
        df = pd.DataFrame()
        result = filter_by_dietary(df, "Any")
        assert result.empty

    def test_empty_dataframe_vegetarian(self):
        df = pd.DataFrame()
        result = filter_by_dietary(df, "Vegetarian")
        assert result.empty

    def test_missing_tags_column(self):
        """DataFrame without 'tags' column falls back to ingredient-only classification."""
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "name": ["Chicken Dish", "Pasta"],
                "ingredients": [
                    "['chicken', 'garlic']",
                    "['pasta', 'tomato']",
                ],
                "minutes": [30, 20],
            }
        )
        result = filter_by_dietary(df, "Non-vegetarian")
        assert len(result) == 1
        assert result.iloc[0]["name"] == "Chicken Dish"

    def test_missing_ingredients_column(self):
        """DataFrame without 'ingredients' column uses tags only."""
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "name": ["Veg Dish", "Beef Dish"],
                "tags": ["['vegetarian']", "['beef']"],
                "minutes": [15, 20],
            }
        )
        result_veg = filter_by_dietary(df, "Vegetarian")
        assert len(result_veg) == 1
        assert result_veg.iloc[0]["name"] == "Veg Dish"

        result_nonveg = filter_by_dietary(df, "Non-vegetarian")
        assert len(result_nonveg) == 1
        assert result_nonveg.iloc[0]["name"] == "Beef Dish"

    def test_missing_both_columns_returns_df_unchanged(self):
        """Without both tags and ingredients, return df unchanged (cannot classify)."""
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"], "minutes": [10, 20]})
        result = filter_by_dietary(df, "Vegetarian")
        assert len(result) == len(df)

    def test_index_reset_after_filter(self):
        """Filtered result has a contiguous 0-based index."""
        df = _make_recipe_df()
        result = filter_by_dietary(df, "Vegetarian")
        assert list(result.index) == list(range(len(result)))


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility — dietary filter + cooking time filter
# ─────────────────────────────────────────────────────────────────────────────

class TestDietaryWithCookingTime:
    """Verify dietary filter and cooking-time filter work together correctly."""

    @staticmethod
    def _cooking_time_filter(df: pd.DataFrame, option: str) -> pd.DataFrame:
        """Inline copy of filter_by_cooking_time for testing without importing app."""
        if df is None or df.empty:
            return pd.DataFrame() if df is None else df
        if option == "Any cooking time" or "minutes" not in df.columns:
            return df
        time_limits = {
            "Up to 15 minutes": 15,
            "Up to 30 minutes": 30,
            "Up to 60 minutes": 60,
        }
        limit = time_limits.get(option)
        if limit is None:
            return df
        minutes_numeric = pd.to_numeric(df["minutes"], errors="coerce")
        valid_mask = (minutes_numeric >= 0) & (minutes_numeric <= limit)
        return df[valid_mask]

    def test_vegetarian_up_to_30_min(self):
        df = _make_recipe_df()
        time_filtered = self._cooking_time_filter(df, "Up to 30 minutes")
        result = filter_by_dietary(time_filtered, "Vegetarian")
        # Veggie Pasta (30 min, veg) ✓ ; Lentil Soup (45 min) filtered out by time
        assert len(result) == 1
        assert result.iloc[0]["name"] == "Veggie Pasta"

    def test_nonveg_up_to_30_min(self):
        df = _make_recipe_df()
        time_filtered = self._cooking_time_filter(df, "Up to 30 minutes")
        result = filter_by_dietary(time_filtered, "Non-vegetarian")
        # Chicken Stir Fry (20 min) ✓ ; Beef Burger (25 min) ✓
        assert len(result) == 2

    def test_any_diet_up_to_15_min(self):
        df = _make_recipe_df()
        time_filtered = self._cooking_time_filter(df, "Up to 15 minutes")
        result = filter_by_dietary(time_filtered, "Any")
        # Only Plain Rice (10 min)
        assert len(result) == 1
        assert result.iloc[0]["name"] == "Plain Rice"

    def test_vegetarian_no_results_in_narrow_time_window(self):
        df = _make_recipe_df()
        time_filtered = self._cooking_time_filter(df, "Up to 15 minutes")
        result = filter_by_dietary(time_filtered, "Vegetarian")
        # Plain Rice (10 min) is uncertain → excluded
        assert len(result) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility — dietary filter with both recommendation methods
# ─────────────────────────────────────────────────────────────────────────────

class TestDietaryWithRecommendationMethods:
    """
    Verify that filter_by_dietary produces consistent results regardless of
    which recommendation method is used downstream.
    The dietary filter operates on the DataFrame *before* ranking, so the
    ranking method (baseline or TF-IDF) must not influence the filter output.
    """

    def test_filtered_df_compatible_with_rank_recipes(self):
        from src.recommender import rank_recipes

        df = _make_recipe_df()
        filtered = filter_by_dietary(df, "Non-vegetarian")
        # rank_recipes must accept the filtered DataFrame without error
        results = rank_recipes(["chicken", "garlic"], filtered, top_n=5)
        assert isinstance(results, list)
        # All results must come from non-vegetarian recipes
        nonveg_names = {"Chicken Stir Fry", "Beef Burger"}
        for r in results:
            assert r.title in nonveg_names or r.match_percentage == 0.0

    def test_filtered_df_compatible_with_rank_recipes_tfidf(self):
        from src.recommender import rank_recipes_tfidf

        df = _make_recipe_df()
        filtered = filter_by_dietary(df, "Vegetarian")
        results = rank_recipes_tfidf(["pasta", "tomato"], filtered, top_n=5)
        assert isinstance(results, list)
        veg_names = {"Veggie Pasta", "Lentil Soup"}
        for r in results:
            assert r.title in veg_names or r.match_percentage == 0.0

    def test_any_preference_does_not_reduce_candidate_pool(self):
        """'Any' must leave all rows available for the recommender."""
        df = _make_recipe_df()
        filtered = filter_by_dietary(df, "Any")
        assert len(filtered) == len(df)
