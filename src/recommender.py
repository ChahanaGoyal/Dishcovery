"""
recommender.py
==============
Ingredient-based recipe recommendation and ranking engine.

Provides baseline rule-based matching: compares the user's available
ingredients against each recipe's ingredients to compute a match percentage,
identifies missing ingredients, and ranks recipes from highest to lowest match.

Match Score Formula
-------------------
    Match percentage = (Available recipe ingredients / Total recipe ingredients) * 100

This is an explainable baseline ingredient-matching metric (not a trained ML model).
It measures how close the user is to having all ingredients required for a dish.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import clean_ingredient_list, parse_ingredient_list

logger = logging.getLogger(__name__)


@dataclass
class Recipe:
    """Container for a single recipe result and its match metrics.

    Attributes
    ----------
    recipe_id:
        Unique identifier for the recipe.
    title:
        Recipe name / title.
    matched_ingredients:
        List of recipe ingredients available to the user.
    score:
        Match score in [0.0, 1.0] (normalized match percentage or cosine similarity).
    match_percentage:
        Match score as a percentage in [0.0, 100.0].
    missing_ingredients:
        List of recipe ingredients the user is missing.
    cooking_time:
        Total preparation and cooking time in minutes (if available).
    similarity_score:
        Cosine similarity score in [0.0, 1.0] when using TF-IDF recommendation.
    """

    recipe_id: str
    title: str
    matched_ingredients: list[str] = field(default_factory=list)
    score: float = 0.0
    match_percentage: float = 0.0
    missing_ingredients: list[str] = field(default_factory=list)
    cooking_time: int | None = None
    similarity_score: float | None = None

    def __post_init__(self) -> None:
        if self.match_percentage == 0.0 and self.score > 0.0 and self.similarity_score is None:
            self.match_percentage = round(self.score * 100.0, 2)
        elif self.score == 0.0 and self.match_percentage > 0.0:
            self.score = round(self.match_percentage / 100.0, 4)
        elif self.similarity_score is not None and self.score == 0.0:
            self.score = round(self.similarity_score, 4)

    @property
    def name(self) -> str:
        """Alias for title to support dataset column naming variations."""
        return self.title

    def to_dict(self) -> dict:
        """Return result as a standard Python dictionary."""
        return {
            "recipe_id": self.recipe_id,
            "title": self.title,
            "match_percentage": self.match_percentage,
            "matched_ingredients": self.matched_ingredients,
            "missing_ingredients": self.missing_ingredients,
            "cooking_time": self.cooking_time,
            "score": self.score,
            "similarity_score": self.similarity_score,
        }


def _parse_and_clean_ingredients(raw: list[str] | str | None) -> list[str]:
    """Extract and normalize ingredients from a list, string, or stringified list."""
    if not raw:
        return []
    if isinstance(raw, list):
        return clean_ingredient_list(raw)
    if isinstance(raw, str):
        val = raw.strip()
        if not val:
            return []
        if val.startswith("[") and val.endswith("]"):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return clean_ingredient_list(parsed)
            except (ValueError, SyntaxError):
                pass
        return parse_ingredient_list(val)
    return []


def calculate_match_score(
    user_ingredients: list[str] | set[str] | None,
    recipe_ingredients: list[str] | str | None,
) -> tuple[float, list[str], list[str]]:
    """Calculate the match percentage, matched ingredients, and missing ingredients.

    Formula:
        Match percentage = (Matched recipe ingredients / Total recipe ingredients) * 100

    Parameters
    ----------
    user_ingredients:
        Ingredients provided by the user.
    recipe_ingredients:
        Ingredients required by the recipe (list, string, or stringified list).

    Returns
    -------
    tuple[float, list[str], list[str]]
        - match_percentage: float between 0.0 and 100.0.
        - matched_ingredients: recipe ingredients present in user input.
        - missing_ingredients: recipe ingredients missing from user input.
    """
    recipe_clean = _parse_and_clean_ingredients(recipe_ingredients)
    if not recipe_clean:
        return 0.0, [], []

    if not user_ingredients:
        return 0.0, [], recipe_clean

    if isinstance(user_ingredients, str):
        user_clean = _parse_and_clean_ingredients(user_ingredients)
        user_set = set(user_clean)
    elif isinstance(user_ingredients, set):
        user_set = {
            i.strip().lower()
            for i in user_ingredients
            if isinstance(i, str) and i.strip()
        }
    else:
        user_set = set(clean_ingredient_list(user_ingredients))

    if not user_set:
        return 0.0, [], recipe_clean

    matched: list[str] = [ing for ing in recipe_clean if ing in user_set]
    missing: list[str] = [ing for ing in recipe_clean if ing not in user_set]

    match_percentage = round((len(matched) / len(recipe_clean)) * 100.0, 2)
    return match_percentage, matched, missing


def match_recipes(
    user_ingredients: list[str] | set[str] | None,
    df: pd.DataFrame,
    min_match: float = 0.0,
) -> list[Recipe]:
    """Calculate match metrics for every recipe in the DataFrame.

    Parameters
    ----------
    user_ingredients:
        Ingredients available to the user.
    df:
        DataFrame containing recipe data.
    min_match:
        Optional minimum match percentage threshold in [0.0, 100.0] (default: 0.0).

    Returns
    -------
    list[Recipe]
        List of Recipe objects containing match percentages and missing ingredients.
    """
    return rank_recipes(user_ingredients, df, top_n=len(df) if df is not None else 0, min_match=min_match)


def rank_recipes(
    user_ingredients: list[str] | set[str] | None,
    df: pd.DataFrame,
    top_n: int = 5,
    min_match: float = 0.0,
) -> list[Recipe]:
    """Rank recipes by ingredient match percentage and return the top matches.

    Recipes are sorted from highest to lowest match percentage. Ties are broken
    consistently by fewest missing ingredients, then alphabetically by title.

    Parameters
    ----------
    user_ingredients:
        Ingredients available to the user.
    df:
        Recipe DataFrame. Supports both Food.com columns ('id', 'name', 'ingredients', 'minutes')
        and preprocessed columns ('recipe_id', 'title', 'ingredients_clean', 'cooking_time').
    top_n:
        Maximum number of recipes to return (default: 5). If <= 0, returns an empty list.
    min_match:
        Minimum match percentage (0.0 to 100.0) required to include a recipe (default: 0.0).

    Returns
    -------
    list[Recipe]
        Ranked list of Recipe results.

    Raises
    ------
    ValueError
        If min_match is not a number between 0.0 and 100.0, or if df lacks an ingredients column.
    """
    if not isinstance(min_match, (int, float)) or not (0.0 <= min_match <= 100.0):
        raise ValueError(
            f"min_match must be a float or int between 0.0 and 100.0, got {min_match!r}"
        )

    if top_n <= 0 or df is None or df.empty:
        return []

    if not user_ingredients:
        return []

    if isinstance(user_ingredients, str):
        user_clean = _parse_and_clean_ingredients(user_ingredients)
    elif isinstance(user_ingredients, set):
        user_clean = [
            i.strip().lower()
            for i in user_ingredients
            if isinstance(i, str) and i.strip()
        ]
    else:
        user_clean = clean_ingredient_list(list(user_ingredients))

    if not user_clean:
        return []

    user_set = set(user_clean)

    # Resolve column names flexibly
    id_col = "recipe_id" if "recipe_id" in df.columns else ("id" if "id" in df.columns else None)
    title_col = "title" if "title" in df.columns else ("name" if "name" in df.columns else None)
    ing_col = (
        "ingredients_clean"
        if "ingredients_clean" in df.columns
        else ("ingredients" if "ingredients" in df.columns else None)
    )
    time_col = "minutes" if "minutes" in df.columns else ("cooking_time" if "cooking_time" in df.columns else None)

    if ing_col is None:
        raise ValueError(
            "DataFrame must contain an ingredients column ('ingredients_clean' or 'ingredients')."
        )

    # Extract columns to Python lists for fast traversal without mutating df
    ids = [str(x) for x in df[id_col].tolist()] if id_col else [str(i) for i in range(len(df))]
    titles = (
        [str(x) if pd.notna(x) else "" for x in df[title_col].tolist()]
        if title_col
        else [f"Recipe {i}" for i in ids]
    )
    raw_ings = df[ing_col].tolist()

    if time_col:
        times: list[int | None] = []
        for val in df[time_col].tolist():
            try:
                times.append(int(val) if pd.notna(val) else None)
            except (ValueError, TypeError):
                times.append(None)
    else:
        times = [None] * len(df)

    candidates: list[Recipe] = []

    for recipe_id, title, raw_ing, cooking_time in zip(ids, titles, raw_ings, times):
        recipe_clean = _parse_and_clean_ingredients(raw_ing)
        if not recipe_clean:
            # Exclude recipes that have no usable ingredient information
            continue

        pct, matched, missing = calculate_match_score(user_set, recipe_clean)
        if pct < min_match:
            continue

        candidates.append(
            Recipe(
                recipe_id=recipe_id,
                title=title,
                matched_ingredients=matched,
                score=round(pct / 100.0, 4),
                match_percentage=pct,
                missing_ingredients=missing,
                cooking_time=cooking_time,
            )
        )

    # Sort descending by match_percentage, tie-break by fewest missing, then title, then recipe_id
    candidates.sort(
        key=lambda r: (-r.match_percentage, len(r.missing_ingredients), r.title, r.recipe_id)
    )

    return candidates[:top_n]


class IngredientRecommender:
    """Recommend recipes based on a list of available ingredients.

    Usage
    -----
    ::
        recommender = IngredientRecommender()
        recommender.fit(recipe_dataframe)
        results = recommender.recommend(["tomato", "garlic", "pasta"], top_n=5)
    """

    def __init__(self) -> None:
        self._is_fitted: bool = False
        self._recipes: pd.DataFrame = pd.DataFrame()

    def fit(self, df: pd.DataFrame) -> "IngredientRecommender":
        """Index the recipe dataset for recommendation.

        Parameters
        ----------
        df:
            Recipe DataFrame. Supports columns ['recipe_id', 'title', 'ingredients_clean']
            or raw Food.com columns ['id', 'name', 'ingredients'].

        Returns
        -------
        IngredientRecommender
            Self, for method chaining.
        """
        has_id = "recipe_id" in df.columns or "id" in df.columns
        has_title = "title" in df.columns or "name" in df.columns
        has_ingredients = "ingredients_clean" in df.columns or "ingredients" in df.columns

        if not (has_id and has_title and has_ingredients):
            raise ValueError(
                "DataFrame must contain an ID ('recipe_id' or 'id'), "
                "title ('title' or 'name'), and ingredients ('ingredients_clean' or 'ingredients'). "
                f"Found columns: {list(df.columns)}"
            )

        self._recipes = df.copy()
        self._is_fitted = True
        logger.info("Recommender fitted on %d recipes.", len(df))
        return self

    def recommend(
        self,
        ingredients: list[str],
        top_n: int = 5,
        min_match: float = 0.0,
    ) -> list[Recipe]:
        """Return the top-N recipes matching the given ingredients.

        Parameters
        ----------
        ingredients:
            List of ingredient names provided by the user.
        top_n:
            Maximum number of results to return.
        min_match:
            Minimum match percentage threshold.

        Returns
        -------
        list[Recipe]
            Ranked list of matching recipes.
        """
        if not self._is_fitted:
            logger.warning("Recommender has not been fitted — returning empty results.")
            return []

        return rank_recipes(ingredients, self._recipes, top_n=top_n, min_match=min_match)

    @property
    def is_fitted(self) -> bool:
        """True once fit() has been called with a valid dataset."""
        return self._is_fitted


# ── TF-IDF / Cosine Similarity Recommendation ────────────────────────────────


def rank_recipes_tfidf(
    user_ingredients: list[str] | str | None,
    df: pd.DataFrame,
    top_n: int = 5,
) -> list[Recipe]:
    """Rank recipes using TF-IDF vectors and cosine similarity.

    Converts each recipe's ingredient list and the user's ingredients into
    TF-IDF representations, then ranks recipes by cosine similarity to the
    user's ingredient query.

    **Score label:** The ``similarity_score`` field holds the raw cosine
    similarity value in [0.0, 1.0]. It is *not* a match percentage — do not
    display it as one.

    Parameters
    ----------
    user_ingredients:
        List of ingredient strings provided by the user, or a comma-separated
        string, or None/empty.
    df:
        Recipe DataFrame. Supports both Food.com columns
        ('id', 'name', 'ingredients', 'minutes') and preprocessed columns
        ('recipe_id', 'title', 'ingredients_clean', 'cooking_time').
    top_n:
        Maximum number of recipes to return (default: 5).

    Returns
    -------
    list[Recipe]
        Recipes ranked by cosine similarity (highest first).
        Each Recipe has ``similarity_score`` populated and
        ``match_percentage`` set to ``similarity_score * 100``.
        The ``matched_ingredients`` and ``missing_ingredients`` fields
        are computed using the same baseline overlap logic for transparency.
    """
    if top_n <= 0 or df is None or df.empty:
        return []

    # Normalise user input
    if not user_ingredients:
        return []
    if isinstance(user_ingredients, str):
        user_clean = _parse_and_clean_ingredients(user_ingredients)
    else:
        user_clean = clean_ingredient_list(list(user_ingredients))

    if not user_clean:
        return []

    # Resolve column names (same logic as rank_recipes)
    id_col = "recipe_id" if "recipe_id" in df.columns else ("id" if "id" in df.columns else None)
    title_col = "title" if "title" in df.columns else ("name" if "name" in df.columns else None)
    ing_col = (
        "ingredients_clean"
        if "ingredients_clean" in df.columns
        else ("ingredients" if "ingredients" in df.columns else None)
    )
    time_col = "minutes" if "minutes" in df.columns else ("cooking_time" if "cooking_time" in df.columns else None)

    if ing_col is None:
        raise ValueError(
            "DataFrame must contain an ingredients column ('ingredients_clean' or 'ingredients')."
        )

    ids = [str(x) for x in df[id_col].tolist()] if id_col else [str(i) for i in range(len(df))]
    titles = (
        [str(x) if pd.notna(x) else "" for x in df[title_col].tolist()]
        if title_col
        else [f"Recipe {i}" for i in ids]
    )
    raw_ings = df[ing_col].tolist()

    if time_col:
        times: list[int | None] = []
        for val in df[time_col].tolist():
            try:
                times.append(int(val) if pd.notna(val) else None)
            except (ValueError, TypeError):
                times.append(None)
    else:
        times = [None] * len(df)

    # Build cleaned ingredient text for each recipe (filter out empties)
    recipe_texts: list[str] = []
    valid_indices: list[int] = []

    for i, raw_ing in enumerate(raw_ings):
        cleaned = _parse_and_clean_ingredients(raw_ing)
        if cleaned:
            recipe_texts.append(" ".join(cleaned))
            valid_indices.append(i)

    if not recipe_texts:
        logger.warning("rank_recipes_tfidf: no recipes with usable ingredient text.")
        return []

    user_text = " ".join(user_clean)

    # Fit TF-IDF on recipe corpus, then transform user query
    try:
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[a-zA-Z]{2,}\b")
        recipe_matrix = vectorizer.fit_transform(recipe_texts)
        user_vector = vectorizer.transform([user_text])
    except ValueError as exc:
        logger.warning("rank_recipes_tfidf: TF-IDF vectorization failed (%s). Returning empty.", exc)
        return []

    sims = cosine_similarity(user_vector, recipe_matrix)[0]  # shape (n_recipes,)

    # Build Recipe objects, including baseline overlap for matched/missing info
    user_set = set(user_clean)
    candidates: list[Recipe] = []

    for rank_idx, orig_idx in enumerate(valid_indices):
        sim = float(sims[rank_idx])
        recipe_ing_clean = _parse_and_clean_ingredients(raw_ings[orig_idx])
        _, matched, missing = calculate_match_score(user_set, recipe_ing_clean)

        candidates.append(
            Recipe(
                recipe_id=ids[orig_idx],
                title=titles[orig_idx],
                matched_ingredients=matched,
                missing_ingredients=missing,
                score=round(sim, 4),
                match_percentage=round(sim * 100.0, 2),
                cooking_time=times[orig_idx],
                similarity_score=round(sim, 4),
            )
        )

    # Sort by similarity descending, break ties alphabetically by title
    candidates.sort(key=lambda r: (-r.similarity_score, r.title, r.recipe_id))  # type: ignore[arg-type]
    return candidates[:top_n]
