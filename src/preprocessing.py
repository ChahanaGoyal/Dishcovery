"""
preprocessing.py
================
Text cleaning and ingredient normalisation utilities.

Transforms raw ingredient strings into a format suitable for
feature extraction and similarity computation.

Public API
----------
- :func:`normalise_ingredient`   – clean a single ingredient string.
- :func:`clean_ingredient_list`  – clean a Python list of ingredient strings
                                   (as returned by ``ast.literal_eval`` on the
                                   Food.com ``ingredients`` column).
- :func:`parse_ingredient_list`  – split a delimited string then clean each part.
- :func:`preprocess_dataframe`   – apply cleaning to a DataFrame column.
"""

from __future__ import annotations

import ast
import re

import pandas as pd


# Characters / patterns to strip from ingredient strings
_PUNCT_RE = re.compile(r"[^\w\s,]")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_ingredient_list(ingredients: list[str] | None) -> list[str]:
    """Clean and deduplicate a list of ingredient strings.

    The function:
    - Converts ingredient strings to lowercase.
    - Removes leading and trailing whitespace.
    - Removes empty strings.
    - Removes duplicate entries while preserving their original order.
    - Handles empty or missing ingredient lists safely.

    Parameters
    ----------
    ingredients:
        List of raw ingredient strings, or None.

    Returns
    -------
    list[str]
        Cleaned and deduplicated list of ingredient strings.
    """
    if not ingredients:
        return []

    cleaned: list[str] = []
    seen: set[str] = set()

    for item in ingredients:
        if not isinstance(item, str):
            continue
        token = item.strip().lower()
        if token and token not in seen:
            seen.add(token)
            cleaned.append(token)

    return cleaned


def normalise_ingredient(ingredient: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace in an ingredient.

    Parameters
    ----------
    ingredient:
        Raw ingredient string, e.g. ``"2 cups All-Purpose Flour!"``.

    Returns
    -------
    str
        Cleaned string, e.g. ``"2 cups allpurpose flour"``.
    """
    text = ingredient.lower()
    text = _PUNCT_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def parse_ingredient_list(raw: str, sep: str = ",") -> list[str]:
    """Split and normalise a delimited ingredient string.

    Parameters
    ----------
    raw:
        Comma-separated (or *sep*-separated) ingredient string.
    sep:
        Delimiter used in *raw*. Defaults to ``","``

    Returns
    -------
    list[str]
        List of normalised ingredient tokens.
    """
    parts = raw.split(sep)
    return [normalise_ingredient(p) for p in parts if p.strip()]


def preprocess_dataframe(df: pd.DataFrame, ingredient_col: str = "ingredients") -> pd.DataFrame:
    """Apply ingredient normalisation to every row of *df*.

    Parameters
    ----------
    df:
        Input DataFrame containing an ingredient column.
    ingredient_col:
        Name of the column holding raw ingredient strings or lists.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with an added ``"ingredients_clean"`` column.

    Raises
    ------
    KeyError
        If *ingredient_col* is not present in *df*.
    """
    if ingredient_col not in df.columns:
        raise KeyError(f"Column '{ingredient_col}' not found in DataFrame.")

    def _process_cell(raw: object) -> list[str]:
        if isinstance(raw, list):
            return clean_ingredient_list(raw)
        if isinstance(raw, str):
            raw_stripped = raw.strip()
            if raw_stripped.startswith("[") and raw_stripped.endswith("]"):
                try:
                    parsed = ast.literal_eval(raw_stripped)
                    if isinstance(parsed, list):
                        return clean_ingredient_list(parsed)
                except (ValueError, SyntaxError):
                    pass
            return parse_ingredient_list(raw)
        return []

    out = df.copy()
    out["ingredients_clean"] = out[ingredient_col].apply(_process_cell)
    return out
