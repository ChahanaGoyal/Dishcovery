"""
recommender.py
==============
Ingredient-based recipe recommendation engine (stub).

The full ML pipeline (TF-IDF vectorisation + cosine similarity) will be
implemented once the dataset is integrated.  This module exposes the public
interface that ``app.py`` and the test suite will program against.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Recipe:
    """Lightweight container for a single recipe result."""

    recipe_id: str
    title: str
    matched_ingredients: list[str] = field(default_factory=list)
    score: float = 0.0


class IngredientRecommender:
    """Recommend recipes based on a list of available ingredients.

    Usage (once a dataset is loaded)::

        recommender = IngredientRecommender()
        recommender.fit(recipe_dataframe)
        results = recommender.recommend(["tomato", "garlic", "pasta"])

    Notes
    -----
    The ``fit`` / ``recommend`` methods are scaffolded here but not yet
    connected to a real dataset or ML model.  They return empty results
    until the data pipeline is complete.
    """

    def __init__(self) -> None:
        self._is_fitted: bool = False
        self._recipes: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "IngredientRecommender":
        """Index the recipe dataset for fast lookup.

        Parameters
        ----------
        df:
            Pre-processed DataFrame with at least ``recipe_id``, ``title``
            and ``ingredients_clean`` columns.

        Returns
        -------
        IngredientRecommender
            Self, for method chaining.
        """
        required = {"recipe_id", "title", "ingredients_clean"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")

        self._recipes = df.copy()
        self._is_fitted = True
        logger.info("Recommender fitted on %d recipes.", len(df))
        return self

    def recommend(
        self,
        ingredients: list[str],
        top_n: int = 5,
    ) -> list[Recipe]:
        """Return the top-N recipes matching the given ingredients.

        Parameters
        ----------
        ingredients:
            List of ingredient names provided by the user.
        top_n:
            Maximum number of results to return.

        Returns
        -------
        list[Recipe]
            Ranked list of matching recipes (empty until dataset is loaded).
        """
        if not self._is_fitted:
            logger.warning("Recommender has not been fitted — returning empty results.")
            return []

        if not ingredients:
            return []

        # ── Placeholder scoring logic ───────────────────────────────────
        # TODO: Replace with TF-IDF cosine similarity once dataset is ready.
        query = {ing.lower().strip() for ing in ingredients}
        results: list[Recipe] = []

        for _, row in self._recipes.iterrows():
            recipe_ings = set(row["ingredients_clean"])
            matched = list(query & recipe_ings)
            score = len(matched) / max(len(query), 1)
            if matched:
                results.append(
                    Recipe(
                        recipe_id=str(row["recipe_id"]),
                        title=str(row["title"]),
                        matched_ingredients=matched,
                        score=round(score, 4),
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_n]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """True once ``fit()`` has been called with a valid dataset."""
        return self._is_fitted
