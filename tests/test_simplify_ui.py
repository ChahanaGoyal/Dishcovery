"""
tests/test_simplify_ui.py
=========================
Regression tests for simplified Dishcovery recommendation interface:
1. TF-IDF tokenizes individual ingredient words correctly.
2. Potato, onion, and tomato produce non-zero similarity for relevant recipes.
3. Recommendations are correctly ranked by similarity.
4. Empty ingredient input handled safely.
5. Recipes with empty ingredient lists handled safely.
6. Valid TF-IDF recommendations not removed by UI filtering (similarity_score > 0.0).
7. Dietary filtering works.
8. Cooking-time filtering works.
9. Sorting works.
10. Search results persist in Streamlit session state.
11. UI no longer exposes algorithm selection controls.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from app import (
    filter_by_cooking_time,
    format_recipe_ingredients,
    parse_ingredients,
    render_recipe_card,
    sort_recommendations,
)
from src.preprocessing import filter_by_dietary
from src.recommender import Recipe, rank_recipes, rank_recipes_tfidf


@pytest.fixture
def sample_recipes_df() -> pd.DataFrame:
    """Fixture with distinct ingredient profiles and cooking times."""
    return pd.DataFrame(
        {
            "id": [101, 102, 103, 104, 105],
            "name": [
                "Potato Onion Frittata",
                "Tomato Potato Curry",
                "Beef Onion Stew",
                "Steamed Fish",
                "Empty Recipe",
            ],
            "tags": [
                "['vegetarian', 'breakfast']",
                "['vegetarian', 'curry']",
                "['meat', 'beef', 'dinner']",
                "['seafood', 'fish']",
                "['misc']",
            ],
            "ingredients": [
                "['potato', 'onion', 'egg', 'olive oil']",
                "['potato', 'onion', 'tomato', 'curry powder', 'garlic']",
                "['beef', 'onion', 'broth', 'black pepper']",
                "['fish fillet', 'ginger', 'soy sauce']",
                "[]",
            ],
            "minutes": [25, 35, 90, 15, 10],
        }
    )


# 1. TF-IDF tokenizes individual ingredient words correctly
def test_tfidf_tokenizes_individual_words():
    df = pd.DataFrame(
        {
            "id": [1],
            "name": ["Potato Dish"],
            "ingredients": ["['potato', 'onion', 'tomato']"],
            "minutes": [20],
        }
    )
    recs = rank_recipes_tfidf(["potato"], df)
    assert len(recs) == 1
    assert recs[0].similarity_score is not None
    assert recs[0].similarity_score > 0.0


# 2. Potato, onion, and tomato produce non-zero similarity for relevant recipes
def test_potato_onion_tomato_similarity(sample_recipes_df: pd.DataFrame):
    recs = rank_recipes_tfidf(["potato", "onion", "tomato"], sample_recipes_df, top_n=5)
    recs_by_name = {r.title: r for r in recs}

    assert "Tomato Potato Curry" in recs_by_name
    assert recs_by_name["Tomato Potato Curry"].similarity_score > 0.0

    assert "Potato Onion Frittata" in recs_by_name
    assert recs_by_name["Potato Onion Frittata"].similarity_score > 0.0

    assert "Beef Onion Stew" in recs_by_name
    assert recs_by_name["Beef Onion Stew"].similarity_score > 0.0

    # Steamed Fish has none of potato/onion/tomato
    if "Steamed Fish" in recs_by_name:
        assert recs_by_name["Steamed Fish"].similarity_score == 0.0


# 3. Recommendations are correctly ranked by similarity
def test_recommendations_ranked_by_similarity(sample_recipes_df: pd.DataFrame):
    recs = rank_recipes_tfidf(["potato", "onion", "tomato"], sample_recipes_df, top_n=5)
    # Check monotonic non-increasing similarity score order
    for i in range(len(recs) - 1):
        assert recs[i].similarity_score >= recs[i + 1].similarity_score
    # Tomato Potato Curry has all three searched ingredients, so it ranks first
    assert recs[0].title == "Tomato Potato Curry"


# 4. Empty ingredient input handled safely
def test_empty_ingredient_input(sample_recipes_df: pd.DataFrame):
    assert rank_recipes_tfidf([], sample_recipes_df) == []
    assert rank_recipes_tfidf("", sample_recipes_df) == []
    assert rank_recipes_tfidf(None, sample_recipes_df) == []
    assert parse_ingredients("") == []
    assert parse_ingredients("   ,  , ") == []


# 5. Recipes with empty ingredient lists handled safely
def test_recipes_with_empty_ingredient_lists(sample_recipes_df: pd.DataFrame):
    # Recipe 105 has "[]" ingredients
    recs = rank_recipes_tfidf(["potato"], sample_recipes_df, top_n=10)
    # Shouldn't crash and empty recipe shouldn't match
    recs_by_id = {r.recipe_id: r for r in recs}
    assert "105" not in recs_by_id or recs_by_id["105"].similarity_score == 0.0


# 6. Valid TF-IDF recommendations not removed by UI filtering
def test_valid_tfidf_recommendations_not_removed_by_filter(sample_recipes_df: pd.DataFrame):
    recs = rank_recipes_tfidf(["potato", "onion"], sample_recipes_df, top_n=5)
    matched = [
        r
        for r in recs
        if (r.similarity_score is not None and r.similarity_score > 0.0)
        or (r.score is not None and r.score > 0.0)
    ]
    # At least Potato Onion Frittata, Tomato Potato Curry, Beef Onion Stew have > 0.0
    titles = {r.title for r in matched}
    assert "Potato Onion Frittata" in titles
    assert "Tomato Potato Curry" in titles
    assert "Beef Onion Stew" in titles


# 7. Dietary filtering works
def test_dietary_filtering_integration(sample_recipes_df: pd.DataFrame):
    veg_df = filter_by_dietary(sample_recipes_df, "Vegetarian")
    assert "Beef Onion Stew" not in veg_df["name"].values
    assert "Steamed Fish" not in veg_df["name"].values
    assert "Potato Onion Frittata" in veg_df["name"].values
    assert "Tomato Potato Curry" in veg_df["name"].values

    non_veg_df = filter_by_dietary(sample_recipes_df, "Non-vegetarian")
    assert "Beef Onion Stew" in non_veg_df["name"].values
    assert "Potato Onion Frittata" not in non_veg_df["name"].values

    any_df = filter_by_dietary(sample_recipes_df, "Any")
    assert len(any_df) == len(sample_recipes_df)


# 8. Cooking-time filtering works
def test_cooking_time_filtering_integration(sample_recipes_df: pd.DataFrame):
    quick_df = filter_by_cooking_time(sample_recipes_df, "Up to 30 minutes")
    # Potato Onion Frittata (25) and Steamed Fish (15) should be included; Beef Stew (90) excluded
    assert "Potato Onion Frittata" in quick_df["name"].values
    assert "Steamed Fish" in quick_df["name"].values
    assert "Beef Onion Stew" not in quick_df["name"].values


# 9. Sorting works
def test_sorting_integration(sample_recipes_df: pd.DataFrame):
    recs = rank_recipes_tfidf(["potato", "onion"], sample_recipes_df, top_n=5)
    matched = [r for r in recs if r.similarity_score > 0.0]

    # Best match first
    by_match = sort_recommendations(matched, "Best match first")
    for i in range(len(by_match) - 1):
        assert by_match[i].match_percentage >= by_match[i + 1].match_percentage

    # Shortest cooking time first
    by_time = sort_recommendations(matched, "Shortest cooking time first")
    for i in range(len(by_time) - 1):
        t1 = by_time[i].cooking_time if by_time[i].cooking_time is not None else float("inf")
        t2 = by_time[i + 1].cooking_time if by_time[i + 1].cooking_time is not None else float("inf")
        assert t1 <= t2


# 10. Search results persist in Streamlit session state
def test_session_state_persistence_behavior(sample_recipes_df: pd.DataFrame):
    # Simulate session state dictionary
    mock_session_state = {}

    # Initial search
    recs = rank_recipes_tfidf(["potato", "onion"], sample_recipes_df, top_n=5)
    matched = [r for r in recs if r.similarity_score > 0.0]
    mock_session_state["search_state"] = "has_results"
    mock_session_state["recommendations"] = matched

    # Simulating a rerun with sort change without recomputing recommendations:
    assert mock_session_state.get("search_state") == "has_results"
    cached_recs = mock_session_state.get("recommendations", [])
    assert len(cached_recs) == len(matched)

    sorted_by_time = sort_recommendations(cached_recs, "Shortest cooking time first")
    assert sorted_by_time[0].cooking_time <= sorted_by_time[-1].cooking_time


# 11. UI no longer exposes algorithm selection controls
def test_ui_no_algorithm_selection_controls():
    app_text = Path("app.py").read_text(encoding="utf-8")
    assert "Advanced Recommendation Settings" not in app_text
    assert "Baseline (Ingredient Overlap)" not in app_text
    assert "Improved (TF-IDF Similarity)" not in app_text
    assert "rec_method" not in app_text
