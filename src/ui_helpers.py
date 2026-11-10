"""
ui_helpers.py
=============
Pure-logic helper functions used by the Dishcovery Streamlit UI.

Placing these in a separate module makes them independently testable
without importing or executing Streamlit.

Public API
----------
- :func:`parse_ingredients`     – parse comma-separated ingredient input.
- :func:`filter_by_cooking_time` – filter a DataFrame by cooking-time option.
- :func:`sort_recommendations`  – sort a list of Recipe results by user choice.
- :func:`format_recipe_ingredients` – format raw recipe ingredients for display.
"""

from __future__ import annotations

import ast

import pandas as pd

from src.recommender import Recipe


def parse_ingredients(raw_input: str) -> list[str]:
    """Parse comma-separated ingredients, stripping whitespace, removing empty items,
    and deduplicating while preserving original order.

    Parameters
    ----------
    raw_input:
        Raw text from the ingredient input box.

    Returns
    -------
    list[str]
        Cleaned, deduplicated ingredient tokens.
    """
    cleaned_ingredients: list[str] = []
    seen: set[str] = set()
    for item in raw_input.split(","):
        cleaned = item.strip()
        if cleaned:
            key = cleaned.lower()
            if key not in seen:
                seen.add(key)
                cleaned_ingredients.append(cleaned)
    return cleaned_ingredients


def filter_by_cooking_time(df: pd.DataFrame, max_time_option: str) -> pd.DataFrame:
    """Filter recipes by cooking time based on the selected UI option.

    Parameters
    ----------
    df:
        DataFrame containing at least a 'minutes' column.
    max_time_option:
        One of 'Any cooking time', 'Up to 15 minutes', 'Up to 30 minutes', 'Up to 60 minutes'.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df

    if max_time_option == "Any cooking time" or "minutes" not in df.columns:
        return df

    time_limits = {
        "Up to 15 minutes": 15,
        "Up to 30 minutes": 30,
        "Up to 60 minutes": 60,
    }
    limit = time_limits.get(max_time_option)
    if limit is None:
        return df

    minutes_numeric = pd.to_numeric(df["minutes"], errors="coerce")
    valid_mask = (minutes_numeric >= 0) & (minutes_numeric <= limit)
    return df[valid_mask]


def sort_recommendations(recipes: list[Recipe], order: str) -> list[Recipe]:
    """Sort a list of Recipe results by the chosen sort order.

    The underlying match/similarity scores are not modified.

    Parameters
    ----------
    recipes:
        List of Recipe results from the recommender.
    order:
        One of:
        - ``'Best match first'``: descending by match_percentage (default).
        - ``'Shortest cooking time first'``: ascending by cooking_time, with
          match_percentage as tie-breaker. Recipes with no recorded cooking
          time are placed at the end.

    Returns
    -------
    list[Recipe]
        A new sorted list; the input list is not mutated.
    """
    if order == "Shortest cooking time first":
        return sorted(
            recipes,
            key=lambda r: (
                r.cooking_time if r.cooking_time is not None else float("inf"),
                -r.match_percentage,  # tie-break: higher match first
            ),
        )
    # Default: best match first
    return sorted(recipes, key=lambda r: -r.match_percentage)


def format_recipe_ingredients(raw: object) -> str:
    """Format raw recipe ingredients from the dataset for display.

    Parameters
    ----------
    raw:
        Value from the ``'ingredients'`` column — may be a Python list,
        a stringified list, a plain string, None, or NaN.

    Returns
    -------
    str
        Comma-separated ingredient string, or ``'Ingredients not available'``.
    """
    if raw is None:
        return "Ingredients not available"
    if isinstance(raw, list):
        items = [str(x).strip() for x in raw if str(x).strip()]
        return ", ".join(items) if items else "Ingredients not available"
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    items = [str(x).strip() for x in parsed if str(x).strip()]
                    return ", ".join(items) if items else "Ingredients not available"
            except (ValueError, SyntaxError):
                pass
        return stripped if stripped else "Ingredients not available"
    try:
        if pd.isna(raw):
            return "Ingredients not available"
    except (TypeError, ValueError):
        pass
    return str(raw)
