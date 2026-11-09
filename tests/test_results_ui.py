"""
tests/test_results_ui.py
========================
Tests for Prompt 3 — Improved recipe results experience:
  - sort_recommendations() — both sort orders, edge cases, tie-breaking, no mutations.
  - render_recipe_card() — score label distinctions (Match vs Similarity),
    matched and missing ingredients styling.
  - Empty input handling and validation.
  - No-results handling.
  - Compatibility with dietary and cooking-time filters.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from app import (
    filter_by_cooking_time,
    format_recipe_ingredients,
    parse_ingredients,
    render_recipe_card,
    sort_recommendations,
)
from src.recommender import Recipe, rank_recipes, rank_recipes_tfidf
from src.preprocessing import filter_by_dietary


# ─────────────────────────────────────────────────────────────────────────────
# Helper for creating Recipe instances
# ─────────────────────────────────────────────────────────────────────────────

def _make_recipe(title: str, pct: float, cooking_time: int | None = None) -> Recipe:
    return Recipe(
        recipe_id=title,
        title=title,
        match_percentage=pct,
        matched_ingredients=["ingredient_a"],
        missing_ingredients=[],
        cooking_time=cooking_time,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sort Recommendations Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSortRecommendations:
    """Unit tests for sort_recommendations()."""

    def test_best_match_first_default(self):
        recs = [
            _make_recipe("Recipe B", 60.0, 30),
            _make_recipe("Recipe A", 90.0, 45),
            _make_recipe("Recipe C", 75.0, 15),
        ]
        result = sort_recommendations(recs, "Best match first")
        assert [r.title for r in result] == ["Recipe A", "Recipe C", "Recipe B"]

    def test_shortest_cooking_time_first(self):
        recs = [
            _make_recipe("Slow Dish", 90.0, 60),
            _make_recipe("Fast Dish", 50.0, 10),
            _make_recipe("Medium Dish", 70.0, 30),
        ]
        result = sort_recommendations(recs, "Shortest cooking time first")
        assert [r.title for r in result] == ["Fast Dish", "Medium Dish", "Slow Dish"]

    def test_shortest_cooking_time_tie_break_by_match(self):
        recs = [
            _make_recipe("Low Match", 50.0, 20),
            _make_recipe("High Match", 90.0, 20),
        ]
        result = sort_recommendations(recs, "Shortest cooking time first")
        assert result[0].title == "High Match"
        assert result[1].title == "Low Match"

    def test_no_cooking_time_placed_at_end(self):
        recs = [
            _make_recipe("No Time Recipe", 95.0, None),
            _make_recipe("Timed Recipe", 60.0, 15),
        ]
        result = sort_recommendations(recs, "Shortest cooking time first")
        assert result[0].title == "Timed Recipe"
        assert result[1].title == "No Time Recipe"

    def test_all_no_cooking_time_sorts_by_match(self):
        recs = [
            _make_recipe("Dish B", 60.0, None),
            _make_recipe("Dish A", 90.0, None),
        ]
        result = sort_recommendations(recs, "Shortest cooking time first")
        assert [r.title for r in result] == ["Dish A", "Dish B"]

    def test_empty_list_returns_empty(self):
        assert sort_recommendations([], "Best match first") == []
        assert sort_recommendations([], "Shortest cooking time first") == []

    def test_single_item_returns_single_item(self):
        recs = [_make_recipe("Only Dish", 55.0, 20)]
        assert sort_recommendations(recs, "Best match first")[0].title == "Only Dish"
        assert sort_recommendations(recs, "Shortest cooking time first")[0].title == "Only Dish"

    def test_scores_and_metrics_not_mutated(self):
        recs = [
            _make_recipe("Dish A", 80.0, 30),
            _make_recipe("Dish B", 60.0, 10),
        ]
        result = sort_recommendations(recs, "Shortest cooking time first")
        result_by_title = {r.title: r for r in result}
        assert result_by_title["Dish A"].match_percentage == 80.0
        assert result_by_title["Dish B"].match_percentage == 60.0
        assert result_by_title["Dish A"].cooking_time == 30
        assert result_by_title["Dish B"].cooking_time == 10

    def test_sorting_compatible_with_baseline_and_tfidf(self):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Pasta", "Rice", "Soup"],
            "ingredients": [
                "['pasta', 'tomato', 'garlic']",
                "['rice', 'water', 'salt']",
                "['chicken', 'water', 'carrot']",
            ],
            "minutes": [30, 10, 45],
        })

        baseline_recs = rank_recipes(["tomato", "garlic"], df, top_n=5)
        sorted_baseline = sort_recommendations(baseline_recs, "Shortest cooking time first")
        assert isinstance(sorted_baseline, list)

        tfidf_recs = rank_recipes_tfidf(["pasta", "tomato"], df, top_n=5)
        sorted_tfidf = sort_recommendations(tfidf_recs, "Best match first")
        assert isinstance(sorted_tfidf, list)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Recipe Card Rendering and Score Labels Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRecipeCardRendering:
    """Verify render_recipe_card score labels and visual distinction."""

    def test_baseline_score_label_and_styling(self):
        rec = _make_recipe("Tomato Pasta", 75.0, 30)
        with patch("app.st.markdown") as mock_markdown:
            render_recipe_card(rec, score_label="Match")
            rendered_html = mock_markdown.call_args[0][0]

            assert "recipe-score-match" in rendered_html
            assert "recipe-score-similarity" not in rendered_html
            assert "Match: 75" in rendered_html
            assert "Tomato Pasta" in rendered_html

    def test_tfidf_score_label_and_styling(self):
        rec = _make_recipe("Garlic Bread", 88.2, 15)
        with patch("app.st.markdown") as mock_markdown:
            render_recipe_card(rec, score_label="Similarity")
            rendered_html = mock_markdown.call_args[0][0]

            assert "recipe-score-similarity" in rendered_html
            assert "recipe-score-match" not in rendered_html
            assert "Similarity: 88.2%" in rendered_html

    def test_matched_and_missing_ingredients_visually_distinct(self):
        rec = Recipe(
            recipe_id="r1",
            title="Veggie Stir Fry",
            match_percentage=66.7,
            matched_ingredients=["broccoli", "bell pepper"],
            missing_ingredients=["sesame oil"],
            cooking_time=15,
        )
        with patch("app.st.markdown") as mock_markdown:
            render_recipe_card(rec, score_label="Match")
            rendered_html = mock_markdown.call_args[0][0]

            # Green class for have, red class for miss
            assert "ing-have" in rendered_html
            assert "broccoli" in rendered_html
            assert "bell pepper" in rendered_html

            assert "ing-miss" in rendered_html
            assert "sesame oil" in rendered_html

    def test_all_ingredients_available_banner(self):
        rec = Recipe(
            recipe_id="r2",
            title="Boiled Egg",
            match_percentage=100.0,
            matched_ingredients=["egg", "water", "salt"],
            missing_ingredients=[],
            cooking_time=8,
        )
        with patch("app.st.markdown") as mock_markdown:
            render_recipe_card(rec, score_label="Match")
            rendered_html = mock_markdown.call_args[0][0]

            assert "ing-all" in rendered_html
            assert "You have all the listed ingredients!" in rendered_html
            assert "ing-miss" not in rendered_html

    def test_cooking_time_display_with_valid_and_missing_time(self):
        rec_timed = _make_recipe("Timed Dish", 50.0, 25)
        rec_untimed = _make_recipe("Untimed Dish", 50.0, None)

        with patch("app.st.markdown") as mock_markdown:
            render_recipe_card(rec_timed, score_label="Match")
            assert "25 mins" in mock_markdown.call_args[0][0]

            render_recipe_card(rec_untimed, score_label="Match")
            assert "Time not specified" in mock_markdown.call_args[0][0]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Empty Input & Validation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptyInputHandling:
    """Verify input parsing properly detects and cleans empty or whitespace inputs."""

    def test_empty_string_returns_empty_list(self):
        assert parse_ingredients("") == []

    def test_whitespace_and_commas_only(self):
        assert parse_ingredients("   ,  , \t\n ") == []

    def test_valid_input_parsed_and_deduplicated(self):
        result = parse_ingredients("potato, onion, tomato, potato, garlic")
        assert result == ["potato", "onion", "tomato", "garlic"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Filter Compatibility & Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestFilterCompatibilityAndEdgeCases:
    """Verify sorting and recommendation flow with dietary and cooking-time filters."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "id": [1, 2, 3, 4],
            "name": ["Veg Soup", "Quick Chicken", "Hearty Beef Stew", "Green Salad"],
            "tags": [
                "['vegetarian', 'soup']",
                "['chicken', 'quick']",
                "['beef', 'meat']",
                "['vegetarian', 'salad']",
            ],
            "ingredients": [
                "['lentils', 'carrot', 'onion']",
                "['chicken', 'garlic', 'oil']",
                "['beef', 'potato', 'carrot']",
                "['lettuce', 'cucumber', 'oil']",
            ],
            "minutes": [45, 15, 90, 10],
        })

    def test_vegetarian_filter_with_cooking_time_and_sorting(self, sample_df):
        # 1. Cooking time up to 60 mins -> Veg Soup (45) and Green Salad (10)
        time_filtered = filter_by_cooking_time(sample_df, "Up to 60 minutes")
        # 2. Vegetarian filter
        diet_filtered = filter_by_dietary(time_filtered, "Vegetarian")
        assert set(diet_filtered["name"]) == {"Veg Soup", "Green Salad"}

        # 3. Recommendations
        recs = rank_recipes(["carrot", "onion", "lettuce", "cucumber"], diet_filtered, top_n=5)
        assert len(recs) == 2

        # 4. Sort by shortest cooking time -> Green Salad (10 mins) first, then Veg Soup (45 mins)
        sorted_time = sort_recommendations(recs, "Shortest cooking time first")
        assert sorted_time[0].title == "Green Salad"
        assert sorted_time[1].title == "Veg Soup"

    def test_no_results_when_all_filtered_out(self, sample_df):
        # Ultra fast non-existent window
        time_filtered = filter_by_cooking_time(sample_df, "Up to 15 minutes")
        diet_filtered = filter_by_dietary(time_filtered, "Vegetarian")
        # Only Green Salad is 10 mins and veg. If user searches for beef:
        recs = rank_recipes(["beef"], diet_filtered, top_n=5)
        # Matches should have 0% match
        matched_recs = [r for r in recs if r.match_percentage > 0.0]
        assert len(matched_recs) == 0

        # sort on empty list produces empty list without error
        sorted_empty = sort_recommendations(matched_recs, "Best match first")
        assert sorted_empty == []
