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
- :func:`classify_dietary`       – classify a recipe row as 'vegetarian',
                                   'non-vegetarian', or 'uncertain'.
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


# ── Dietary classification ─────────────────────────────────────────────────────

#: Food.com tag substrings that reliably indicate a non-vegetarian recipe.
_NONVEG_TAGS: frozenset[str] = frozenset(
    {
        "beef",
        "pork",
        "chicken",
        "turkey",
        "lamb",
        "bacon",
        "ham",
        "fish",
        "seafood",
        "shellfish",
        "shrimp",
        "crab",
        "lobster",
        "poultry",
        "meat",
        "duck",
        "venison",
        "goat",
        "bison",
        "veal",
        "anchovies",
        "tuna",
        "salmon",
        "sardines",
    }
)

#: Ingredient keyword substrings that indicate a non-vegetarian recipe.
#: Words are matched as substrings of lowercased ingredient tokens.
_NONVEG_INGREDIENT_KEYWORDS: tuple[str, ...] = (
    "beef",
    "chicken",
    "pork",
    "bacon",
    "ham",
    "turkey",
    "lamb",
    "salmon",
    "tuna",
    "shrimp",
    "crab",
    "lobster",
    "anchov",
    "sardine",
    "clam",
    "mussel",
    "oyster",
    "scallop",
    "squid",
    "fish sauce",
    "worcestershire",  # traditionally contains anchovies
    "lard",
    "gelatin",
    "duck",
    "venison",
    "pepperoni",
    "salami",
    "prosciutto",
    "sausage",
    "chorizo",
    "meatball",
    "ground meat",
    "minced meat",
    "veal",
    "bison",
)


def classify_dietary(
    tags: list[str] | str | None,
    ingredients: list[str] | str | None,
) -> str:
    """Classify a recipe as 'vegetarian', 'non-vegetarian', or 'uncertain'.

    Classification logic (conservative — avoids false vegetarian labels):

    1. If the ``tags`` column is available and contains a recognised non-vegetarian
       tag, the recipe is classified as **'non-vegetarian'**.
    2. If ``tags`` contains the ``'vegetarian'`` tag *and* no non-vegetarian tags
       are present, it is classified as **'vegetarian'**.
    3. If ``ingredients`` contains any non-vegetarian keyword, the recipe is
       classified as **'non-vegetarian'**.
    4. If none of the above match, the recipe is classified as **'uncertain'**
       rather than 'vegetarian', to avoid incorrect labelling.

    Parameters
    ----------
    tags:
        Recipe tags — either a Python list or a stringified list from the
        Food.com ``tags`` column.  ``None`` or empty is handled safely.
    ingredients:
        Recipe ingredients — list or stringified list.  ``None`` or empty is
        handled safely.

    Returns
    -------
    str
        One of ``'vegetarian'``, ``'non-vegetarian'``, or ``'uncertain'``.
    """
    # ── Parse tags ────────────────────────────────────────────────────────────
    parsed_tags: list[str] = []
    if tags is not None:
        if isinstance(tags, list):
            parsed_tags = [str(t).strip().lower() for t in tags if t]
        elif isinstance(tags, str):
            stripped = tags.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    result = ast.literal_eval(stripped)
                    if isinstance(result, list):
                        parsed_tags = [str(t).strip().lower() for t in result if t]
                except (ValueError, SyntaxError):
                    pass
            elif stripped:
                parsed_tags = [stripped.lower()]

    tag_set: set[str] = set(parsed_tags)

    # ── Check non-veg tags ────────────────────────────────────────────────────
    if tag_set & _NONVEG_TAGS:
        return "non-vegetarian"

    # ── Check vegetarian tag (only if no non-veg tags matched) ────────────────
    has_veg_tag = "vegetarian" in tag_set

    # ── Parse ingredients ─────────────────────────────────────────────────────
    parsed_ingredients: list[str] = []
    if ingredients is not None:
        if isinstance(ingredients, list):
            parsed_ingredients = [str(i).strip().lower() for i in ingredients if i]
        elif isinstance(ingredients, str):
            stripped = ingredients.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    result = ast.literal_eval(stripped)
                    if isinstance(result, list):
                        parsed_ingredients = [str(i).strip().lower() for i in result if i]
                except (ValueError, SyntaxError):
                    pass
            elif stripped:
                parsed_ingredients = [stripped.lower()]

    combined_text = " ".join(parsed_ingredients)

    # ── Check non-veg ingredient keywords ────────────────────────────────────
    for keyword in _NONVEG_INGREDIENT_KEYWORDS:
        if keyword in combined_text:
            return "non-vegetarian"

    # ── Resolve classification ────────────────────────────────────────────────
    if has_veg_tag:
        return "vegetarian"

    # No positive vegetarian signal — classify as uncertain rather than
    # assuming vegetarian (conservative approach).
    return "uncertain"



def filter_by_dietary(df: "pd.DataFrame", preference: str) -> "pd.DataFrame":
    """Filter a recipe DataFrame by dietary preference.

    Uses :func:`classify_dietary` on each row's tags and ingredients.
    Classification results:

    - ``'vegetarian'``: recipe has the vegetarian tag, no non-veg tags/ingredients.
    - ``'non-vegetarian'``: recipe has meat/poultry/seafood tags or ingredients.
    - ``'uncertain'``: classification could not be determined confidently.

    Parameters
    ----------
    df:
        Recipe DataFrame with at least ``'tags'`` and/or ``'ingredients'`` columns.
    preference:
        One of ``'Any'``, ``'Vegetarian'``, ``'Non-vegetarian'``.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.  When *preference* is ``'Any'`` or columns are
        missing, the original DataFrame is returned unchanged.

    Notes
    -----
    Recipes with uncertain classification are excluded from both 'Vegetarian'
    and 'Non-vegetarian' filters (conservative approach).
    """
    if df is None or df.empty or preference == "Any":
        return df

    target = "vegetarian" if preference == "Vegetarian" else "non-vegetarian"
    if "dietary_class" in df.columns:
        return df[df["dietary_class"] == target].reset_index(drop=True)

    has_tags = "tags" in df.columns
    has_ingredients = "ingredients" in df.columns

    if not has_tags and not has_ingredients:
        # Cannot classify without either column; return unchanged.
        return df

    tags_col = df["tags"].tolist() if has_tags else [None] * len(df)
    ings_col = df["ingredients"].tolist() if has_ingredients else [None] * len(df)

    classifications = [
        classify_dietary(t, i)
        for t, i in zip(tags_col, ings_col)
    ]

    import numpy as np
    bool_mask = np.array([c == target for c in classifications], dtype=bool)
    return df.iloc[np.where(bool_mask)[0]].reset_index(drop=True)
