"""
preprocessing.py
================
Text cleaning and ingredient normalisation utilities.

Transforms raw ingredient strings into a format suitable for
feature extraction and similarity computation.
"""

from __future__ import annotations

import re

import pandas as pd


# Characters / patterns to strip from ingredient strings
_PUNCT_RE = re.compile(r"[^\w\s,]")
_WHITESPACE_RE = re.compile(r"\s+")


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
        Name of the column holding raw ingredient strings.

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

    out = df.copy()
    out["ingredients_clean"] = out[ingredient_col].apply(
        lambda raw: parse_ingredient_list(str(raw))
    )
    return out
