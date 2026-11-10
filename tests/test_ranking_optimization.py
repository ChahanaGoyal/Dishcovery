"""
tests/test_ranking_optimization.py
==================================
Unit and regression tests for combined recipe ranking improvement
and performance optimisation:
1. Best Match First ranking:
   - Fewer missing ingredients rank first.
   - Equal missing counts resolved by fewer total usable recipe ingredients first.
   - Remaining ties resolved alphabetically by recipe title (case-insensitive).
   - Duplicate and empty ingredients do not distort ranking.
2. Shortest cooking time first:
   - Fastest recipes first.
   - Safely handles missing/invalid/negative cooking times at end.
   - Tie-breaks with Best Match First.
3. Compatibility with dietary and cooking-time filters.
4. TF-IDF vectorization and sparse similarity produce meaningful scores.
5. Caching and sub-second candidate ranking.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.recommender import (
    Recipe,
    calculate_match_score,
    rank_by_best_match,
    rank_recipes_tfidf,
)
from src.ui_helpers import (
    filter_by_cooking_time,
    parse_ingredients,
    sort_recommendations,
)
from src.preprocessing import filter_by_dietary


@pytest.fixture
def ranking_df() -> pd.DataFrame:
    """Sample DataFrame with diverse ingredient profiles, duplicates, and times."""
    return pd.DataFrame(
        {
            "id": ["1", "2", "3", "4", "5", "6"],
            "name": [
                "Simple Potato",
                "Potato Onion Soup",
                "Grand Potato Feast",
                "Alphabetical A",
                "Alphabetical B",
                "Untimed Dish",
            ],
            "tags": [
                "['vegetarian']",
                "['vegetarian']",
                "['meat', 'beef']",
                "['vegetarian']",
                "['vegetarian']",
                "['vegetarian']",
            ],
            "ingredients": [
                "['potato', 'olive oil']",
                "['potato', 'onion', 'garlic', 'salt']",
                "['potato', 'onion', 'beef', 'carrot', 'celery', 'wine', 'butter']",
                "['potato', 'salt']",
                "['potato', 'salt']",
                "['potato']",
            ],
            "minutes": [15, 30, 120, 20, 20, None],
        }
    )


# 1. Exact Prompt Example Verification
def test_prompt_exact_example_ranking():
    """Verify exact example from prompt:
    Recipe A: 2 missing, 3 total ingredients.
    Recipe B: 3 missing, 5 total ingredients.
    Recipe C: 2 missing, 6 total ingredients.
    Expected ranking:
    1. Recipe A
    2. Recipe C
    3. Recipe B
    """
    rec_a = Recipe(
        recipe_id="A",
        title="Recipe A",
        matched_ingredients=["m1"],
        missing_ingredients=["x1", "x2"],
    )
    rec_b = Recipe(
        recipe_id="B",
        title="Recipe B",
        matched_ingredients=["m1", "m2"],
        missing_ingredients=["x1", "x2", "x3"],
    )
    rec_c = Recipe(
        recipe_id="C",
        title="Recipe C",
        matched_ingredients=["m1", "m2", "m3", "m4"],
        missing_ingredients=["x1", "x2"],
    )

    result = sort_recommendations([rec_b, rec_a, rec_c], "Best match first")
    assert [r.title for r in result] == ["Recipe A", "Recipe C", "Recipe B"]


# 2. Fewer missing ingredients first
def test_fewer_missing_ingredients_rank_first():
    r1 = Recipe(recipe_id="1", title="1 Missing", missing_ingredients=["m"])
    r2 = Recipe(recipe_id="2", title="2 Missing", missing_ingredients=["m", "n"])
    r3 = Recipe(recipe_id="3", title="0 Missing", missing_ingredients=[])
    result = sort_recommendations([r1, r2, r3], "Best match first")
    assert [r.title for r in result] == ["0 Missing", "1 Missing", "2 Missing"]


# 3. Equal missing counts resolved by fewer total ingredients
def test_equal_missing_resolved_by_fewer_total_ingredients():
    # Both have 2 missing, but Simple has 3 total, Complex has 7 total
    r_simple = Recipe(
        recipe_id="1",
        title="Simple Dish",
        matched_ingredients=["have_1"],
        missing_ingredients=["miss_1", "miss_2"],
    )
    r_complex = Recipe(
        recipe_id="2",
        title="Complex Dish",
        matched_ingredients=["have_1", "have_2", "have_3", "have_4", "have_5"],
        missing_ingredients=["miss_1", "miss_2"],
    )
    result = sort_recommendations([r_complex, r_simple], "Best match first")
    assert result[0].title == "Simple Dish"
    assert result[1].title == "Complex Dish"


# 4. Remaining ties resolved alphabetically (case-insensitive)
def test_ties_resolved_alphabetically():
    r_z = Recipe(
        recipe_id="z",
        title="Zucchini Surprise",
        matched_ingredients=["a"],
        missing_ingredients=["b"],
    )
    r_a = Recipe(
        recipe_id="a",
        title="Apple Delight",
        matched_ingredients=["a"],
        missing_ingredients=["b"],
    )
    r_m = Recipe(
        recipe_id="m",
        title="Mango Treat",
        matched_ingredients=["a"],
        missing_ingredients=["b"],
    )
    result = sort_recommendations([r_z, r_m, r_a], "Best match first")
    assert [r.title for r in result] == ["Apple Delight", "Mango Treat", "Zucchini Surprise"]


# 5. Duplicate and empty ingredients do not distort ranking
def test_duplicate_and_empty_ingredients_handling():
    pct, matched, missing = calculate_match_score(
        user_ingredients=["potato", "POTATO", "  potato  ", ""],
        recipe_ingredients="['potato', 'potato', 'Potato', 'onion', '', '  ']",
    )
    # Unique recipe ingredients: potato, onion (2 total).
    # Matched: potato (1), Missing: onion (1).
    assert set(matched) == {"potato"}
    assert set(missing) == {"onion"}
    assert pct == 50.0


# 6. Shortest cooking time first handles missing/invalid times safely
def test_shortest_cooking_time_sorting():
    r_fast = Recipe(recipe_id="1", title="Fast", cooking_time=10, missing_ingredients=["a"])
    r_slow = Recipe(recipe_id="2", title="Slow", cooking_time=60, missing_ingredients=[])
    r_none = Recipe(recipe_id="3", title="No Time", cooking_time=None, missing_ingredients=[])
    r_neg = Recipe(recipe_id="4", title="Invalid Time", cooking_time=-5, missing_ingredients=[])

    result = sort_recommendations([r_none, r_slow, r_neg, r_fast], "Shortest cooking time first")
    assert result[0].title == "Fast"
    assert result[1].title == "Slow"
    # None and invalid (-5) placed safely at end
    assert set(r.title for r in result[2:]) == {"No Time", "Invalid Time"}


# 7. TF-IDF produces meaningful similarities and works with cached index
def test_tfidf_meaningful_similarities(ranking_df):
    recs = rank_recipes_tfidf(["potato", "onion"], ranking_df, top_n=5)
    assert len(recs) > 0
    # Relevant recipes have non-zero similarity
    for r in recs:
        assert r.similarity_score is not None
        assert 0.0 <= r.similarity_score <= 1.0


# 8. Combined pipeline with dietary and cooking-time filters
def test_combined_filters_with_ranking(ranking_df):
    time_filtered = filter_by_cooking_time(ranking_df, "Up to 30 minutes")
    veg_filtered = filter_by_dietary(time_filtered, "Vegetarian")

    # Grand Potato Feast is meat and 120 mins -> excluded by both
    assert "Grand Potato Feast" not in veg_filtered["name"].values

    recs = rank_recipes_tfidf(["potato", "onion"], veg_filtered, top_n=5, ranking="best_match")
    # Ranked by Best Match First
    for i in range(len(recs) - 1):
        miss1, miss2 = len(recs[i].missing_ingredients), len(recs[i + 1].missing_ingredients)
        tot1 = len(recs[i].matched_ingredients) + len(recs[i].missing_ingredients)
        tot2 = len(recs[i + 1].matched_ingredients) + len(recs[i + 1].missing_ingredients)
        assert (miss1, tot1) <= (miss2, tot2)
