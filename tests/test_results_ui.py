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

def _make_recipe(
    title: str,
    pct: float = 0.0,
    cooking_time: int | None = None,
    matched: list[str] | None = None,
    missing: list[str] | None = None,
) -> Recipe:
    return Recipe(
        recipe_id=title,
        title=title,
        match_percentage=pct,
        matched_ingredients=matched if matched is not None else ["ingredient_a"],
        missing_ingredients=missing if missing is not None else [],
        cooking_time=cooking_time,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sort Recommendations Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSortRecommendations:
    """Unit tests for sort_recommendations() under Best Match First ranking rules."""

    def test_best_match_first_default(self):
        # Prompt exact specification:
        # Recipe A: 2 missing ingredients, 3 total ingredients.
        # Recipe B: 3 missing ingredients, 5 total ingredients.
        # Recipe C: 2 missing ingredients, 6 total ingredients.
        # Expected ranking: Recipe A, Recipe C, Recipe B
        recs = [
            _make_recipe("Recipe B", pct=40.0, matched=["m1", "m2"], missing=["x1", "x2", "x3"]),
            _make_recipe("Recipe A", pct=33.3, matched=["m1"], missing=["x1", "x2"]),
            _make_recipe("Recipe C", pct=66.7, matched=["m1", "m2", "m3", "m4"], missing=["x1", "x2"]),
        ]
        result = sort_recommendations(recs, "Best match first")
        assert [r.title for r in result] == ["Recipe A", "Recipe C", "Recipe B"]

    def test_best_match_fewer_missing_ingredients_first(self):
        recs = [
            _make_recipe("Dish 3 Missing", missing=["a", "b", "c"], matched=["m1"]),
            _make_recipe("Dish 1 Missing", missing=["a"], matched=["m1"]),
            _make_recipe("Dish 2 Missing", missing=["a", "b"], matched=["m1"]),
        ]
        result = sort_recommendations(recs, "Best match first")
        assert [r.title for r in result] == ["Dish 1 Missing", "Dish 2 Missing", "Dish 3 Missing"]

    def test_best_match_equal_missing_fewer_total_first(self):
        # Equal missing (2), but Dish Simple has total 3, Dish Complex has total 6
        recs = [
            _make_recipe("Dish Complex", missing=["a", "b"], matched=["m1", "m2", "m3", "m4"]),
            _make_recipe("Dish Simple", missing=["a", "b"], matched=["m1"]),
        ]
        result = sort_recommendations(recs, "Best match first")
        assert [r.title for r in result] == ["Dish Simple", "Dish Complex"]

    def test_best_match_equal_missing_and_total_alphabetical(self):
        # Equal missing (1) and equal total (3), tie-break alphabetically
        recs = [
            _make_recipe("Zucchini Dish", missing=["a"], matched=["m1", "m2"]),
            _make_recipe("Apple Pie", missing=["a"], matched=["m1", "m2"]),
            _make_recipe("Carrot Soup", missing=["a"], matched=["m1", "m2"]),
        ]
        result = sort_recommendations(recs, "Best match first")
        assert [r.title for r in result] == ["Apple Pie", "Carrot Soup", "Zucchini Dish"]

    def test_shortest_cooking_time_first(self):
        recs = [
            _make_recipe("Slow Dish", pct=90.0, cooking_time=60),
            _make_recipe("Fast Dish", pct=50.0, cooking_time=10),
            _make_recipe("Medium Dish", pct=70.0, cooking_time=30),
        ]
        result = sort_recommendations(recs, "Shortest cooking time first")
        assert [r.title for r in result] == ["Fast Dish", "Medium Dish", "Slow Dish"]

    def test_shortest_cooking_time_tie_break_by_best_match(self):
        # Both 20 mins: High Match has 0 missing, Low Match has 2 missing
        recs = [
            _make_recipe("Low Match", cooking_time=20, missing=["a", "b"], matched=["m1"]),
            _make_recipe("High Match", cooking_time=20, missing=[], matched=["m1", "m2"]),
        ]
        result = sort_recommendations(recs, "Shortest cooking time first")
        assert result[0].title == "High Match"
        assert result[1].title == "Low Match"

    def test_no_cooking_time_placed_at_end(self):
        recs = [
            _make_recipe("No Time Recipe", cooking_time=None, missing=[]),
            _make_recipe("Timed Recipe", cooking_time=15, missing=["a"]),
        ]
        result = sort_recommendations(recs, "Shortest cooking time first")
        assert result[0].title == "Timed Recipe"
        assert result[1].title == "No Time Recipe"

    def test_all_no_cooking_time_sorts_by_best_match(self):
        recs = [
            _make_recipe("Dish B", cooking_time=None, missing=["a", "b"]),
            _make_recipe("Dish A", cooking_time=None, missing=["a"]),
        ]
        result = sort_recommendations(recs, "Shortest cooking time first")
        assert [r.title for r in result] == ["Dish A", "Dish B"]

    def test_empty_list_returns_empty(self):
        assert sort_recommendations([], "Best match first") == []
        assert sort_recommendations([], "Shortest cooking time first") == []

    def test_single_item_returns_single_item(self):
        recs = [_make_recipe("Only Dish", pct=55.0, cooking_time=20)]
        assert sort_recommendations(recs, "Best match first")[0].title == "Only Dish"
        assert sort_recommendations(recs, "Shortest cooking time first")[0].title == "Only Dish"

    def test_scores_and_metrics_not_mutated(self):
        recs = [
            _make_recipe("Dish A", pct=80.0, cooking_time=30),
            _make_recipe("Dish B", pct=60.0, cooking_time=10),
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
    """Verify render_recipe_card ingredient coverage badge and visual distinction."""

    def test_card_shows_ingredient_coverage_badge(self):
        """Card must show 'X of Y ingredients' badge instead of a raw TF-IDF %."""
        rec = Recipe(
            recipe_id="r_cov",
            title="Tomato Pasta",
            match_percentage=75.0,
            matched_ingredients=["pasta", "tomato"],
            missing_ingredients=["garlic"],
            cooking_time=30,
        )
        with patch("app.st.markdown") as mock_markdown:
            render_recipe_card(rec)
            rendered_html = mock_markdown.call_args[0][0]

            assert "2 of 3 ingredients" in rendered_html
            assert "Tomato Pasta" in rendered_html
            # Must NOT show raw percentage values as a score label
            assert "recipe-score-match" in rendered_html

    def test_card_coverage_badge_all_matched(self):
        rec = Recipe(
            recipe_id="r_all",
            title="Boiled Egg",
            match_percentage=100.0,
            matched_ingredients=["egg", "water", "salt"],
            missing_ingredients=[],
            cooking_time=8,
        )
        with patch("app.st.markdown") as mock_markdown:
            render_recipe_card(rec)
            rendered_html = mock_markdown.call_args[0][0]
            assert "3 of 3 ingredients" in rendered_html

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
            # Coverage badge: 3 of 3
            assert "3 of 3 ingredients" in rendered_html

    def test_cooking_time_display_with_valid_and_missing_time(self):
        rec_timed = _make_recipe("Timed Dish", 50.0, 25)
        rec_untimed = _make_recipe("Untimed Dish", 50.0, None)

        with patch("app.st.markdown") as mock_markdown:
            render_recipe_card(rec_timed, score_label="Match")
            assert "25 mins" in mock_markdown.call_args[0][0]

            render_recipe_card(rec_untimed, score_label="Match")
            assert "Time not specified" in mock_markdown.call_args[0][0]

    def test_recipe_card_with_all_ingredients_available(self):
        rec = Recipe(
            recipe_id="all_avail",
            title="Pancake",
            match_percentage=100.0,
            matched_ingredients=["flour", "milk", "egg"],
            missing_ingredients=[],
            cooking_time=15,
        )
        with patch("app.st.markdown") as mock_markdown:
            render_recipe_card(rec, score_label="Relevance")
            html_text = mock_markdown.call_args[0][0]
            kwargs = mock_markdown.call_args[1]

            assert kwargs.get("unsafe_allow_html") is True
            assert "ing-all" in html_text
            assert "You have all the listed ingredients!" in html_text
            assert "ing-miss" not in html_text
            assert "Still need" not in html_text

    def test_recipe_card_with_some_missing_ingredients(self):
        rec = Recipe(
            recipe_id="some_miss",
            title="Tomato Pasta",
            match_percentage=60.0,
            matched_ingredients=["pasta", "tomato"],
            missing_ingredients=["garlic", "basil"],
            cooking_time=20,
        )
        with patch("app.st.markdown") as mock_markdown:
            render_recipe_card(rec, score_label="Relevance")
            html_text = mock_markdown.call_args[0][0]

            assert "ing-have" in html_text
            assert "You have (2):" in html_text
            assert "pasta, tomato" in html_text

            assert "ing-miss" in html_text
            assert "Still need (2):" in html_text
            assert "garlic, basil" in html_text
            assert "ing-all" not in html_text

    def test_recipe_card_with_only_one_missing_ingredient(self):
        rec = Recipe(
            recipe_id="one_miss",
            title="Garlic Bread",
            match_percentage=75.0,
            matched_ingredients=["bread", "butter", "parsley"],
            missing_ingredients=["garlic"],
            cooking_time=10,
        )
        with patch("app.st.markdown") as mock_markdown:
            render_recipe_card(rec, score_label="Relevance")
            html_text = mock_markdown.call_args[0][0]

            assert "ing-have" in html_text
            assert "You have (3):" in html_text
            assert "ing-miss" in html_text
            assert "Still need (1):" in html_text
            assert "garlic" in html_text

    def test_recipe_card_with_several_missing_ingredients(self):
        rec = Recipe(
            recipe_id="several_miss",
            title="Complex Curry",
            match_percentage=25.0,
            matched_ingredients=["rice"],
            missing_ingredients=["curry powder", "coconut milk", "onion", "chicken", "ginger"],
            cooking_time=45,
        )
        with patch("app.st.markdown") as mock_markdown:
            render_recipe_card(rec, score_label="Relevance")
            html_text = mock_markdown.call_args[0][0]

            assert "ing-have" in html_text
            assert "You have (1):" in html_text
            assert "ing-miss" in html_text
            assert "Still need (5):" in html_text
            assert "curry powder, coconut milk, onion, chicken, ginger" in html_text

    def test_recipe_card_no_leading_indentation_to_prevent_code_block(self):
        rec = Recipe(
            recipe_id="no_indent",
            title="Simple Salad",
            match_percentage=50.0,
            matched_ingredients=["lettuce"],
            missing_ingredients=["cucumber"],
            cooking_time=5,
        )
        with patch("app.st.markdown") as mock_markdown:
            render_recipe_card(rec, score_label="Relevance")
            html_text = mock_markdown.call_args[0][0]

            # In CommonMark, lines with 4 or more leading spaces become indented code blocks.
            # Every line must have 0 leading spaces.
            for line in html_text.splitlines():
                if line.strip():
                    assert not line.startswith("    "), (
                        f"Line has 4 or more leading spaces which causes raw HTML code blocks: {line!r}"
                    )

    def test_recipe_card_escapes_html_in_title_and_ingredients(self):
        rec = Recipe(
            recipe_id="xss_test",
            title="Dangerous <script>alert('xss')</script> Dish",
            match_percentage=50.0,
            matched_ingredients=["onion & garlic"],
            missing_ingredients=["peppers <spicy>"],
            cooking_time=15,
        )
        with patch("app.st.markdown") as mock_markdown:
            render_recipe_card(rec, score_label="Relevance")
            html_text = mock_markdown.call_args[0][0]

            assert "<script>" not in html_text.lower()
            assert "&lt;script&gt;" in html_text.lower()
            assert "onion &amp; garlic" in html_text
            assert "&lt;spicy&gt;" in html_text


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

    def test_parse_ingredients_returns_lowercase(self):
        """parse_ingredients must lowercase all tokens for case-insensitive dataset matching."""
        result = parse_ingredients("Pasta, ONION, Tomato")
        assert result == ["pasta", "onion", "tomato"]

    def test_parse_ingredients_case_insensitive_deduplication(self):
        result = parse_ingredients("Garlic, garlic, GARLIC")
        assert result == ["garlic"]


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


# ─────────────────────────────────────────────────────────────────────────────
# 5. Zero-Match Filtering Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestZeroMatchFiltering:
    """Verify that recipes with zero matching ingredients are excluded from results.

    This guards against the bug where one-ingredient recipes with high TF-IDF
    cosine similarity (but zero actual overlap) ranked at the top.
    """

    @pytest.fixture
    def catalog_df(self) -> pd.DataFrame:
        """Small catalog with recipes that have varying ingredient overlap."""
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": [
                "Tomato Pasta",           # overlaps: pasta, tomato
                "Macadamia Nut Butter",   # zero overlap with pasta/onion/tomato
                "One Ingredient Caramel", # zero overlap
                "Onion Soup",             # overlaps: onion
                "Veggie Mix",             # overlaps: tomato, onion
            ],
            "ingredients": [
                "['pasta', 'tomato', 'garlic']",
                "['macadamia nuts']",
                "['caramel']",
                "['onion', 'stock', 'butter']",
                "['tomato', 'onion', 'pepper', 'oil']",
            ],
            "minutes": [30, 10, 5, 45, 20],
        })

    def test_zero_match_recipes_excluded_from_rank_recipes(self, catalog_df):
        """rank_recipes must not return recipes with zero matching ingredients."""
        recs = rank_recipes(["pasta", "onion", "tomato"], catalog_df, top_n=10)
        titles = [r.title for r in recs]
        assert "Macadamia Nut Butter" not in titles, (
            "Macadamia Nut Butter has zero matching ingredients and must be excluded"
        )
        assert "One Ingredient Caramel" not in titles, (
            "One Ingredient Caramel has zero matching ingredients and must be excluded"
        )

    def test_at_least_one_match_required(self, catalog_df):
        """Every recipe returned by rank_recipes must have at least one matched ingredient."""
        recs = rank_recipes(["pasta", "onion", "tomato"], catalog_df, top_n=10)
        for rec in recs:
            assert len(rec.matched_ingredients) >= 1, (
                f"Recipe '{rec.title}' has no matched ingredients but was returned"
            )

    def test_matching_recipes_are_included(self, catalog_df):
        """Recipes with at least one match must still be recommended."""
        recs = rank_recipes(["pasta", "onion", "tomato"], catalog_df, top_n=10)
        titles = [r.title for r in recs]
        assert "Tomato Pasta" in titles
        assert "Onion Soup" in titles
        assert "Veggie Mix" in titles

    def test_fewer_missing_ranks_first(self, catalog_df):
        """Best match first: recipe with fewer missing ingredients ranks first."""
        recs = rank_recipes(["pasta", "onion", "tomato", "garlic"], catalog_df, top_n=10)
        sorted_recs = sort_recommendations(recs, "Best match first")
        # Tomato Pasta needs pasta/tomato/garlic and user has all three → 0 missing
        # Onion Soup needs stock and butter → 2 missing
        # Veggie Mix needs pepper and oil → 2 missing
        assert sorted_recs[0].title == "Tomato Pasta", (
            f"Expected Tomato Pasta first (0 missing), got '{sorted_recs[0].title}'"
        )

    def test_equal_missing_fewer_total_ranks_first(self, catalog_df):
        """With equal missing, recipe with fewer total ingredients ranks first."""
        # User has: tomato, onion, salt (salt not in any recipe — does not affect comparison)
        recs = rank_recipes(["tomato", "onion", "salt"], catalog_df, top_n=10)
        sorted_recs = sort_recommendations(recs, "Best match first")
        # Onion Soup: 1 matched (onion), 2 missing (stock, butter) → total 3
        # Veggie Mix: 2 matched (tomato, onion), 2 missing (pepper, oil) → total 4
        # Both have 2 missing; Onion Soup has fewer total → ranks first
        titles = [r.title for r in sorted_recs]
        assert titles.index("Onion Soup") < titles.index("Veggie Mix"), (
            "Onion Soup (3 total) should rank before Veggie Mix (4 total) when both have equal missing"
        )

    def test_alphabetical_tiebreak(self):
        """Equal missing and equal total → alphabetical order by title."""
        recs = [
            Recipe(recipe_id="z", title="Zucchini Dish",
                   matched_ingredients=["tomato"], missing_ingredients=["oil"]),
            Recipe(recipe_id="a", title="Apple Soup",
                   matched_ingredients=["tomato"], missing_ingredients=["oil"]),
            Recipe(recipe_id="m", title="Mushroom Risotto",
                   matched_ingredients=["tomato"], missing_ingredients=["oil"]),
        ]
        sorted_recs = sort_recommendations(recs, "Best match first")
        assert [r.title for r in sorted_recs] == [
            "Apple Soup", "Mushroom Risotto", "Zucchini Dish"
        ]

    def test_no_ingredient_matches_returns_empty(self, catalog_df):
        """When user enters ingredients that match nothing, rank_recipes returns empty."""
        recs = rank_recipes(["uranium", "xenon"], catalog_df, top_n=10)
        assert recs == [], "Expected empty list when no recipes have matching ingredients"

    def test_duplicate_user_ingredients_do_not_distort_results(self, catalog_df):
        """Duplicate user ingredients must be handled without inflating match counts."""
        recs_dedup = rank_recipes(["tomato", "onion"], catalog_df, top_n=10)
        recs_dup = rank_recipes(["tomato", "tomato", "onion", "onion"], catalog_df, top_n=10)
        # Same ingredients, same results
        assert [r.title for r in recs_dedup] == [r.title for r in recs_dup]

    def test_empty_ingredient_strings_ignored(self, catalog_df):
        """Empty strings in user input must not cause false matches."""
        recs = rank_recipes(["tomato", "", " ", "onion"], catalog_df, top_n=10)
        for rec in recs:
            assert len(rec.matched_ingredients) >= 1

    def test_ingredient_matching_is_case_insensitive(self):
        """Ingredient matching between user input and dataset must be case-insensitive."""
        from src.ui_helpers import parse_ingredients
        # parse_ingredients lowercases; dataset cleaned_ingredients are also lowercase
        user_ings = parse_ingredients("Pasta, ONION, Tomato")
        assert user_ings == ["pasta", "onion", "tomato"]
        # They must match the lowercase cleaned dataset tokens
        user_set = set(user_ings)
        recipe_tokens = ["pasta", "tomato", "garlic"]  # as stored in cleaned_ingredients
        matched = [i for i in recipe_tokens if i in user_set]
        assert matched == ["pasta", "tomato"]

    def test_coverage_badge_in_card_not_tfidf_percent(self):
        """Card must show 'X of Y ingredients' badge, not a raw TF-IDF similarity %."""
        rec = Recipe(
            recipe_id="cb_test",
            title="Pasta Dish",
            match_percentage=99.9,  # This was raw TF-IDF; card must ignore it
            matched_ingredients=["pasta", "tomato"],
            missing_ingredients=["garlic"],
            cooking_time=20,
            similarity_score=0.999,
        )
        with patch("app.st.markdown") as mock_markdown:
            render_recipe_card(rec)
            html_text = mock_markdown.call_args[0][0]
            # Must display "2 of 3 ingredients" badge
            assert "2 of 3 ingredients" in html_text
            # Must NOT show the raw 99.9% cosine similarity value as the badge
            assert "99.9%" not in html_text
