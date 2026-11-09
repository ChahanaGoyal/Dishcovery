"""
app.py – Dishcovery Streamlit Application
==========================================
Entry point for the Dishcovery web app.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import get_data_dir, load_csv, validate_foodcom_columns
from src.preprocessing import clean_ingredient_list
from src.recommender import Recipe, rank_recipes, rank_recipes_tfidf

logger = logging.getLogger(__name__)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dishcovery",
    page_icon="🍽️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* ── Hero ───────────────────────────────────────────── */
        .hero-title {
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #FF6B35 0%, #F7931E 50%, #FFD700 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            margin-bottom: 0.25rem;
            line-height: 1.1;
        }
        .hero-tagline {
            font-size: 1.15rem;
            color: #9CA3AF;
            text-align: center;
            margin-bottom: 0;
            font-style: italic;
        }

        /* ── Section headings ───────────────────────────────── */
        .section-label {
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #F7931E;
            margin-bottom: 0.4rem;
            margin-top: 1.5rem;
        }

        /* ── Recommendation result heading ──────────────────── */
        .results-heading {
            font-size: 1.1rem;
            font-weight: 700;
            color: #F7931E;
            margin-top: 0.25rem;
            margin-bottom: 1rem;
        }

        /* ── Recipe card ────────────────────────────────────── */
        .recipe-card {
            background: #1E1E2E;
            border: 1px solid #2D2D3F;
            border-left: 4px solid #F7931E;
            border-radius: 10px;
            padding: 1rem 1.25rem 0.75rem 1.25rem;
            margin-bottom: 1rem;
        }
        .recipe-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #F9FAFB;
            margin-bottom: 0.15rem;
        }
        .recipe-meta {
            font-size: 0.82rem;
            color: #9CA3AF;
            margin-bottom: 0.6rem;
        }
        .recipe-score {
            font-size: 0.85rem;
            font-weight: 700;
            color: #FFD700;
        }
        .ing-row {
            font-size: 0.88rem;
            margin-bottom: 0.3rem;
            line-height: 1.5;
        }
        .ing-have  { color: #34D399; }
        .ing-miss  { color: #F87171; }
        .ing-all   { color: #34D399; font-style: italic; }
        .ing-label { font-weight: 600; }

        /* ── Catalog card ───────────────────────────────────── */
        .catalog-card {
            background: #1E1E2E;
            border: 1px solid #2D2D3F;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.6rem;
        }
        .catalog-title {
            font-size: 0.95rem;
            font-weight: 600;
            color: #F9FAFB;
        }
        .catalog-meta {
            font-size: 0.82rem;
            color: #9CA3AF;
            margin-top: 0.1rem;
        }

        /* ── Footer ─────────────────────────────────────────── */
        .dishcovery-footer {
            text-align: center;
            color: #6B7280;
            font-size: 0.82rem;
            margin-top: 2.5rem;
            padding-top: 1rem;
            border-top: 1px solid #2D2D3F;
        }

        /* ── Streamlit widget tweaks ─────────────────────────── */
        /* Widen the radio label so the two options breathe */
        div[data-testid="stRadio"] label {
            font-size: 0.92rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Hero section ──────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🍽️ Dishcovery</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-tagline">'
    "Turn what's in your fridge into tonight's dinner — "
    "<em>no grocery run required.</em>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ── Ingredient input ──────────────────────────────────────────────────────────
st.markdown('<div class="section-label">🥕 Your ingredients</div>', unsafe_allow_html=True)

ingredient_input = st.text_area(
    label="Ingredients",
    placeholder="e.g.  tomato, garlic, pasta, olive oil, basil",
    height=110,
    label_visibility="collapsed",
    help="Enter your available ingredients separated by commas.",
)

# ── Controls row: cooking time + method ──────────────────────────────────────
col_time, col_method = st.columns([1, 2], gap="large")

with col_time:
    st.markdown('<div class="section-label">⏱️ Max cooking time</div>', unsafe_allow_html=True)
    cooking_time = st.selectbox(
        label="Cooking time",
        options=["Any cooking time", "Up to 15 minutes", "Up to 30 minutes", "Up to 60 minutes"],
        index=0,
        label_visibility="collapsed",
        help="Filter recipes by how long they take to prepare.",
    )

with col_method:
    st.markdown('<div class="section-label">🤖 Recommendation method</div>', unsafe_allow_html=True)
    rec_method = st.radio(
        label="Recommendation method",
        options=["Baseline (Ingredient Overlap)", "Improved (TF-IDF Similarity)"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        help=(
            "Baseline counts how many recipe ingredients you have. "
            "TF-IDF uses text similarity to find recipes whose ingredient profiles are closest to yours."
        ),
    )

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1.5, 1, 1.5])
with col2:
    find_clicked = st.button("✨ Find Recipes", use_container_width=True, type="primary")


# ── Helper functions ──────────────────────────────────────────────────────────

def parse_ingredients(raw_input: str) -> list[str]:
    """Parse comma-separated ingredients, stripping whitespace, removing empty items,
    and deduplicating while preserving original order.
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


def format_recipe_ingredients(raw: object) -> str:
    """Format raw recipe ingredients from the dataset for display."""
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
    if pd.isna(raw):
        return "Ingredients not available"
    return str(raw)


@st.cache_data
def load_recipe_catalog() -> tuple[pd.DataFrame | None, str | None]:
    """Load the recipe dataset using the existing data loader with caching."""
    data_path = get_data_dir("raw") / "RAW_recipes.csv"
    if not data_path.exists():
        return None, (
            f"Recipe dataset not found at `{data_path}`. "
            "Please download the Food.com `RAW_recipes.csv` and place it in `data/raw/`."
        )
    try:
        df = load_csv(data_path)
        validate_foodcom_columns(df)
        return df, None
    except Exception as exc:
        return None, f"Error loading recipe dataset: {exc}"


def render_recipe_card(recipe: Recipe, score_label: str = "Match") -> None:
    """Render a single recipe recommendation card using styled HTML."""
    title_str = (
        str(recipe.title).title()
        if recipe.title and str(recipe.title).strip()
        else "Untitled Recipe"
    )
    time_str = (
        f"⏱️ {recipe.cooking_time} mins"
        if recipe.cooking_time is not None
        else "⏱️ Time not specified"
    )
    match_pct = round(recipe.match_percentage, 1)

    total_req = recipe.matched_ingredients + recipe.missing_ingredients
    req_html = (
        f'<div class="ing-row"><span class="ing-label">📋 Ingredients needed:</span> '
        f'{", ".join(total_req)}</div>'
        if total_req
        else ""
    )

    if recipe.matched_ingredients:
        have_html = (
            f'<div class="ing-row ing-have"><span class="ing-label">'
            f'✅ You have ({len(recipe.matched_ingredients)}):</span> '
            f'{", ".join(recipe.matched_ingredients)}</div>'
        )
    else:
        have_html = ""

    if recipe.missing_ingredients:
        miss_html = (
            f'<div class="ing-row ing-miss"><span class="ing-label">'
            f'🛒 Still need ({len(recipe.missing_ingredients)}):</span> '
            f'{", ".join(recipe.missing_ingredients)}</div>'
        )
    else:
        miss_html = '<div class="ing-row ing-all">🎉 You have all the listed ingredients!</div>'

    st.markdown(
        f"""
        <div class="recipe-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div class="recipe-title">🍽️ {title_str}</div>
                <div class="recipe-score">🎯 {score_label}: {match_pct}%</div>
            </div>
            <div class="recipe-meta">{time_str}</div>
            {req_html}
            {have_html}
            {miss_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Result area ───────────────────────────────────────────────────────────────
st.divider()
st.markdown('<div class="results-heading">📋 Recipe Recommendations</div>', unsafe_allow_html=True)

if find_clicked:
    cleaned_ingredients = parse_ingredients(ingredient_input)

    if not cleaned_ingredients:
        st.warning("Please enter at least one ingredient before searching.")
    else:
        recipes_df, load_err = load_recipe_catalog()

        if load_err:
            st.error(load_err)
        elif recipes_df is None or recipes_df.empty:
            st.error(
                "Recipe dataset could not be loaded or is empty. "
                "Please verify that `data/raw/RAW_recipes.csv` is present and valid."
            )
        else:
            time_label = cooking_time if cooking_time != "Any cooking time" else "any duration"
            use_tfidf = rec_method.startswith("Improved")
            method_label = "TF-IDF Similarity" if use_tfidf else "Ingredient Overlap"
            st.info(
                f"**Searching with ingredients ({len(cleaned_ingredients)}):** "
                + ", ".join(f"`{i}`" for i in cleaned_ingredients)
                + f" &nbsp;·&nbsp; **Max cooking time:** {time_label}"
                + f" &nbsp;·&nbsp; **Method:** {method_label}",
                icon="🔍",
            )

            try:
                filtered_df = filter_by_cooking_time(recipes_df, cooking_time)

                if filtered_df.empty:
                    st.info(
                        "No matching recipes found. "
                        "Try adding more ingredients or increasing your cooking-time limit."
                    )
                else:
                    if use_tfidf:
                        recommendations = rank_recipes_tfidf(
                            user_ingredients=cleaned_ingredients,
                            df=filtered_df,
                            top_n=5,
                        )
                        score_label = "Similarity"
                    else:
                        recommendations = rank_recipes(
                            user_ingredients=cleaned_ingredients,
                            df=filtered_df,
                            top_n=5,
                            min_match=0.0,
                        )
                        score_label = "Match"

                    matched_recs = [r for r in recommendations if r.match_percentage > 0.0]

                    if not matched_recs:
                        st.info(
                            "No matching recipes found. "
                            "Try adding more ingredients or increasing your cooking-time limit."
                        )
                    else:
                        st.markdown(
                            f"**Top {len(matched_recs)} recipe(s) — {method_label}:**"
                        )
                        for rec in matched_recs:
                            render_recipe_card(rec, score_label=score_label)

            except Exception as exc:
                logger.error("Error during recipe recommendation: %s", exc, exc_info=True)
                st.error("An unexpected error occurred while generating recommendations. Please try again.")
else:
    st.markdown(
        "<p style='color:#6B7280; font-style:italic; margin-top:0.5rem;'>"
        "Enter your ingredients above and click <strong>✨ Find Recipes</strong> to discover matching dishes."
        "</p>",
        unsafe_allow_html=True,
    )

# ── Recipe Catalog Preview (collapsed by default) ─────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("🍲 Browse Recipe Catalog", expanded=False):
    recipes_df, load_err = load_recipe_catalog()

    if load_err:
        st.warning(load_err)
    elif recipes_df is not None and not recipes_df.empty:
        filtered_recipes = filter_by_cooking_time(recipes_df, cooking_time)

        st.caption(
            f"Showing up to 5 sample recipes · filter: **{cooking_time}** "
            f"({len(filtered_recipes):,} matching in catalog)"
        )

        if filtered_recipes.empty:
            st.info("No recipes found within this cooking-time limit.")
        else:
            sample_to_show = filtered_recipes.head(5)
            for _, row in sample_to_show.iterrows():
                name = row.get("name")
                recipe_title = (
                    str(name).title()
                    if pd.notna(name) and str(name).strip()
                    else "Untitled Recipe"
                )

                raw_min = row.get("minutes")
                if pd.notna(raw_min):
                    try:
                        min_val = float(raw_min)
                        time_str = (
                            f"{int(min_val)} mins"
                            if min_val.is_integer()
                            else f"{min_val} mins"
                        )
                    except (ValueError, TypeError):
                        time_str = f"{raw_min} mins"
                else:
                    time_str = "Time not specified"

                ing_str = format_recipe_ingredients(row.get("ingredients"))

                st.markdown(
                    f"""
                    <div class="catalog-card">
                        <div class="catalog-title">🍽️ {recipe_title}
                            <span style="font-weight:400; color:#9CA3AF; font-size:0.82rem;">
                            &nbsp;· ⏱️ {time_str}</span>
                        </div>
                        <div class="catalog-meta"><strong>Ingredients:</strong> {ing_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("Dataset not loaded — see setup instructions in the README.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='dishcovery-footer'>"
    "Dishcovery &nbsp;·&nbsp; Baseline ingredient matching &amp; TF-IDF similarity"
    "&nbsp;·&nbsp; Built with Streamlit &amp; ❤️"
    "</div>",
    unsafe_allow_html=True,
)
