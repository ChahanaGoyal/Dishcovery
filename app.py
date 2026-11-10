"""
app.py – Dishcovery Streamlit Application
==========================================
Entry point for the Dishcovery web app.

Run with:
    streamlit run app.py

Features:
  - Clean, user-friendly recipe search from kitchen ingredients.
  - Automatic TF-IDF cosine similarity recommendation engine.
  - Dietary Preference selector (Any / Vegetarian / Non-vegetarian).
  - Maximum cooking-time selector.
  - Sort results by Best match or Shortest cooking time.
  - Search results persist across Streamlit reruns via session state.
"""

from __future__ import annotations

import ast
import html
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_loader import get_data_dir, load_csv, validate_foodcom_columns
from src.preprocessing import clean_ingredient_list, classify_dietary, filter_by_dietary
from src.recommender import (
    Recipe,
    _parse_and_clean_ingredients,
    rank_by_best_match,
    rank_recipes,
    rank_recipes_tfidf,
)
from src.ui_helpers import (
    filter_by_cooking_time,
    format_recipe_ingredients,
    parse_ingredients,
    sort_recommendations,
)

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

        /* ── Ingredient suggestion hint ─────────────────────── */
        .ing-suggestion {
            font-size: 0.82rem;
            color: #6B7280;
            margin-top: 0.3rem;
            font-style: italic;
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
        /* Baseline match score — orange/amber */
        .recipe-score-match {
            font-size: 0.85rem;
            font-weight: 700;
            color: #FFD700;
        }
        /* TF-IDF similarity score — blue-green */
        .recipe-score-similarity {
            font-size: 0.85rem;
            font-weight: 700;
            color: #67E8F9;
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

# ── Ingredient suggestion hint ────────────────────────────────────────────────
st.markdown(
    '<div class="ing-suggestion">'
    "💡 Try: <em>potato, onion, tomato, garlic</em> — or <em>chicken, rice, soy sauce, ginger</em>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Controls row: cooking time + dietary ─────────────────────────────────────
col_time, col_diet = st.columns([1, 1], gap="large")

with col_time:
    st.markdown('<div class="section-label">⏱️ Max cooking time</div>', unsafe_allow_html=True)
    cooking_time = st.selectbox(
        label="Cooking time",
        options=["Any cooking time", "Up to 15 minutes", "Up to 30 minutes", "Up to 60 minutes"],
        index=0,
        label_visibility="collapsed",
        help="Filter recipes by how long they take to prepare.",
    )

with col_diet:
    st.markdown('<div class="section-label">🥗 Dietary preference</div>', unsafe_allow_html=True)
    dietary_pref = st.radio(
        label="Dietary preference",
        options=["Any", "Vegetarian", "Non-vegetarian"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        help=(
            "Any: show all recipes. "
            "Vegetarian: recipes with no meat, poultry, or seafood. "
            "Non-vegetarian: recipes containing meat, poultry, or seafood."
        ),
    )

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1.5, 1, 1.5])
with col2:
    find_clicked = st.button("✨ Find Recipes", use_container_width=True, type="primary")


@st.cache_data
def load_recipe_catalog() -> tuple[pd.DataFrame | None, str | None]:
    """Load the recipe dataset using the existing data loader with caching and precomputed columns."""
    data_path = get_data_dir("raw") / "RAW_recipes.csv"
    if not data_path.exists():
        return None, (
            f"Recipe dataset not found at `{data_path}`. "
            "Please download the Food.com `RAW_recipes.csv` and place it in `data/raw/`."
        )
    try:
        df = load_csv(data_path)
        validate_foodcom_columns(df)

        # Precompute preprocessed ingredients and text representation once
        raw_ings = df["ingredients"].tolist()
        cleaned_ings = [_parse_and_clean_ingredients(x) for x in raw_ings]
        df["cleaned_ingredients"] = cleaned_ings
        df["ingredient_text"] = [" ".join(x) for x in cleaned_ings]

        # Precompute conservative dietary classification if tags are available
        if "tags" in df.columns:
            tags_list = df["tags"].tolist()
            df["dietary_class"] = [
                classify_dietary(t, ing) for t, ing in zip(tags_list, raw_ings)
            ]
        return df, None
    except Exception as exc:
        return None, f"Error loading recipe dataset: {exc}"


@st.cache_resource
def get_tfidf_index(_df: pd.DataFrame) -> tuple[TfidfVectorizer, Any]:
    """Fit and cache TF-IDF vectorizer and sparse matrix on recipe texts."""
    texts = _df["ingredient_text"].tolist()
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[a-zA-Z]{2,}\b")
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def render_recipe_card(recipe: Recipe, score_label: str = "Relevance") -> None:
    """Render a single recipe recommendation card using styled HTML.

    Uses distinct colours for Match/Relevance % vs Similarity %
    so users are never confused about which score they are seeing.
    All user and recipe strings are safely HTML-escaped to prevent injection.
    Lines contain zero indentation to prevent markdown parsers from treating
    the HTML as an indented code block.
    """
    raw_title = (
        str(recipe.title).title()
        if recipe.title and str(recipe.title).strip()
        else "Untitled Recipe"
    )
    safe_title = html.escape(raw_title)

    time_str = (
        f"⏱️ {recipe.cooking_time} mins"
        if recipe.cooking_time is not None
        else "⏱️ Time not specified"
    )
    safe_time = html.escape(time_str)

    if recipe.similarity_score is not None and recipe.similarity_score > 0.0:
        match_pct = round(recipe.similarity_score * 100.0, 1)
    else:
        match_pct = round(recipe.match_percentage, 1)

    # Score label class: amber for Match/Relevance, cyan for Similarity
    score_css_class = (
        "recipe-score-similarity" if score_label == "Similarity" else "recipe-score-match"
    )
    score_icon = "🔵" if score_label == "Similarity" else "🎯"
    safe_score_label = html.escape(score_label)

    total_req = recipe.matched_ingredients + recipe.missing_ingredients
    safe_total_req = [html.escape(str(x)) for x in total_req]
    safe_matched = [html.escape(str(x)) for x in recipe.matched_ingredients]
    safe_missing = [html.escape(str(x)) for x in recipe.missing_ingredients]

    req_html = (
        f'<div class="ing-row"><span class="ing-label">📋 Ingredients needed:</span> '
        f'{", ".join(safe_total_req)}</div>\n'
        if total_req
        else ""
    )

    if recipe.matched_ingredients:
        have_html = (
            f'<div class="ing-row ing-have"><span class="ing-label">'
            f'✅ You have ({len(recipe.matched_ingredients)}):</span> '
            f'{", ".join(safe_matched)}</div>\n'
        )
    else:
        have_html = ""

    if recipe.missing_ingredients:
        miss_html = (
            f'<div class="ing-row ing-miss"><span class="ing-label">'
            f'🛒 Still need ({len(recipe.missing_ingredients)}):</span> '
            f'{", ".join(safe_missing)}</div>\n'
        )
    else:
        miss_html = '<div class="ing-row ing-all">🎉 You have all the listed ingredients!</div>\n'

    card_html = (
        f'<div class="recipe-card">\n'
        f'<div style="display:flex; justify-content:space-between; align-items:flex-start;">\n'
        f'<div class="recipe-title">🍽️ {safe_title}</div>\n'
        f'<div class="{score_css_class}">{score_icon} {safe_score_label}: {match_pct}%</div>\n'
        f'</div>\n'
        f'<div class="recipe-meta">{safe_time}</div>\n'
        f'{req_html}'
        f'{have_html}'
        f'{miss_html}'
        f'</div>'
    )

    st.markdown(card_html, unsafe_allow_html=True)


# ── Result area ───────────────────────────────────────────────────────────────
st.divider()

if find_clicked:
    cleaned_ingredients = parse_ingredients(ingredient_input)

    if not cleaned_ingredients:
        st.session_state["search_state"] = "empty_input"
        st.session_state["recommendations"] = []
    else:
        recipes_df, load_err = load_recipe_catalog()

        if load_err:
            st.session_state["search_state"] = "error"
            st.session_state["search_error_msg"] = load_err
            st.session_state["recommendations"] = []
        elif recipes_df is None or recipes_df.empty:
            st.session_state["search_state"] = "error"
            st.session_state["search_error_msg"] = (
                "Recipe dataset could not be loaded or is empty. "
                "Please verify that `data/raw/RAW_recipes.csv` is present and valid."
            )
            st.session_state["recommendations"] = []
        else:
            try:
                # 1. Inexpensive filters first
                filtered_df = filter_by_cooking_time(recipes_df, cooking_time)
                filtered_df = filter_by_dietary(filtered_df, dietary_pref)

                if filtered_df.empty:
                    st.session_state["search_state"] = "no_filter_matches"
                    st.session_state["recommendations"] = []
                else:
                    # 2. Cached TF-IDF query transformation & sparse similarity
                    vectorizer, recipe_matrix = get_tfidf_index(recipes_df)

                    user_text = " ".join(cleaned_ingredients)
                    user_vec = vectorizer.transform([user_text])

                    filtered_indices = filtered_df.index.to_numpy()
                    sub_matrix = recipe_matrix[filtered_indices]
                    sims = sub_matrix.dot(user_vec.T).toarray().ravel()

                    pos_mask = sims > 0.0
                    if not np.any(pos_mask):
                        st.session_state["search_state"] = "no_ingredient_matches"
                        st.session_state["recommendations"] = []
                    else:
                        cand_indices = filtered_indices[pos_mask]
                        cand_sims = sims[pos_mask]
                        cand_df = filtered_df.loc[cand_indices]

                        titles = cand_df["name"].tolist()
                        ids = cand_df["id"].tolist()
                        times = cand_df["minutes"].tolist()
                        c_ings = cand_df["cleaned_ingredients"].tolist()

                        user_set = set(cleaned_ingredients)
                        matched_recs = []
                        for r_id, title, raw_time, sim, ing_list in zip(
                            ids, titles, times, cand_sims, c_ings
                        ):
                            matched = [i for i in ing_list if i in user_set]
                            missing = [i for i in ing_list if i not in user_set]
                            try:
                                c_time = (
                                    int(raw_time)
                                    if (raw_time is not None and pd.notna(raw_time) and float(raw_time) >= 0)
                                    else None
                                )
                            except (ValueError, TypeError):
                                c_time = None

                            matched_recs.append(
                                Recipe(
                                    recipe_id=str(r_id),
                                    title=str(title) if pd.notna(title) else "",
                                    matched_ingredients=matched,
                                    missing_ingredients=missing,
                                    cooking_time=c_time,
                                    score=round(float(sim), 4),
                                    similarity_score=round(float(sim), 4),
                                    match_percentage=round(float(sim) * 100.0, 2),
                                )
                            )

                        # 3. Default Best Match First ranking
                        matched_recs = sort_recommendations(matched_recs, "Best match first")

                        st.session_state["search_state"] = "has_results"
                        st.session_state["recommendations"] = matched_recs

                st.session_state["search_ingredients"] = cleaned_ingredients
                st.session_state["search_cooking_time"] = cooking_time
                st.session_state["search_dietary_pref"] = dietary_pref
            except Exception as exc:
                logger.error("Error during recipe recommendation: %s", exc, exc_info=True)
                st.session_state["search_state"] = "error"
                st.session_state["search_error_msg"] = (
                    "Something went wrong while searching for recipes. "
                    "Please try again or adjust your filters."
                )
                st.session_state["recommendations"] = []

# Render results from session state
search_state = st.session_state.get("search_state")

if search_state is None:
    st.markdown('<div class="results-heading">📋 Recipe Recommendations</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#6B7280; font-style:italic; margin-top:0.5rem;'>"
        "Enter your ingredients above and click <strong>✨ Find Recipes</strong> to discover matching dishes."
        "</p>",
        unsafe_allow_html=True,
    )
elif search_state == "empty_input":
    st.markdown('<div class="results-heading">📋 Recipe Recommendations</div>', unsafe_allow_html=True)
    st.warning(
        "⚠️ **No ingredients entered.** "
        "Please type at least one ingredient in the box above, then click **✨ Find Recipes**. "
        "\n\n💡 *Example: tomato, garlic, pasta, olive oil*"
    )
elif search_state == "error":
    st.markdown('<div class="results-heading">📋 Recipe Recommendations</div>', unsafe_allow_html=True)
    st.error(st.session_state.get("search_error_msg", "An unexpected error occurred."))
elif search_state == "no_filter_matches":
    st.markdown('<div class="results-heading">📋 Recipe Recommendations</div>', unsafe_allow_html=True)
    st.info(
        "🔍 **No recipes matched your filters.**\n\n"
        "Try one of these:\n"
        "- Add more ingredients\n"
        "- Change the cooking time to a longer limit\n"
        "- Switch dietary preference to **Any**"
    )
elif search_state == "no_ingredient_matches":
    st.markdown('<div class="results-heading">📋 Recipe Recommendations</div>', unsafe_allow_html=True)
    st.info(
        "🔍 **No recipes matched your ingredients.**\n\n"
        "Try one of these:\n"
        "- Add more ingredients (e.g. common pantry items like oil, salt, garlic)\n"
        "- Change the cooking time to a longer limit\n"
        "- Switch dietary preference to **Any**"
    )
elif search_state == "has_results":
    recs = st.session_state.get("recommendations", [])
    search_ings = st.session_state.get("search_ingredients", [])
    search_time = st.session_state.get("search_cooking_time", cooking_time)
    search_diet = st.session_state.get("search_dietary_pref", dietary_pref)
    time_label = search_time if search_time != "Any cooking time" else "any duration"

    res_col1, res_col2 = st.columns([1.6, 1.4], gap="medium")
    with res_col1:
        st.markdown('<div class="results-heading">📋 Recipe Recommendations</div>', unsafe_allow_html=True)
    with res_col2:
        sort_order = st.selectbox(
            label="Sort results by",
            options=["Best match first", "Shortest cooking time first"],
            index=0,
            key="sort_order_select",
            label_visibility="collapsed",
            help="Choose whether to sort by best match percentage or shortest cooking time.",
        )

    st.info(
        f"**Ingredients ({len(search_ings)}):** "
        + ", ".join(f"`{i}`" for i in search_ings)
        + f" &nbsp;·&nbsp; **Max time:** {time_label}"
        + f" &nbsp;·&nbsp; **Diet:** {search_diet}",
        icon="🔍",
    )

    sorted_recs = sort_recommendations(recs, sort_order)[:5]
    sort_note = (
        "⏱️ fastest first"
        if sort_order == "Shortest cooking time first"
        else "🎯 best match first"
    )
    st.markdown(f"**Top {len(sorted_recs)} recommended recipe(s) ({sort_note}):**")
    for rec in sorted_recs:
        render_recipe_card(rec, score_label="Relevance")

# ── Recipe Catalog Preview (collapsed by default) ─────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("🍲 Browse Recipe Catalog", expanded=False):
    recipes_df, load_err = load_recipe_catalog()

    if load_err:
        st.warning(load_err)
    elif recipes_df is not None and not recipes_df.empty:
        filtered_recipes = filter_by_cooking_time(recipes_df, cooking_time)
        filtered_recipes = filter_by_dietary(filtered_recipes, dietary_pref)

        st.caption(
            f"Showing up to 5 sample recipes · filter: **{cooking_time}** · diet: **{dietary_pref}** "
            f"({len(filtered_recipes):,} matching in catalog)"
        )

        if filtered_recipes.empty:
            st.info("No recipes found with the current cooking-time and dietary filters.")
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

                safe_cat_title = html.escape(recipe_title)
                safe_cat_time = html.escape(time_str)
                safe_cat_ing = html.escape(ing_str)

                catalog_html = (
                    f'<div class="catalog-card">\n'
                    f'<div class="catalog-title">🍽️ {safe_cat_title} '
                    f'<span style="font-weight:400; color:#9CA3AF; font-size:0.82rem;">'
                    f'&nbsp;· ⏱️ {safe_cat_time}</span>\n'
                    f'</div>\n'
                    f'<div class="catalog-meta"><strong>Ingredients:</strong> {safe_cat_ing}</div>\n'
                    f'</div>'
                )

                st.markdown(catalog_html, unsafe_allow_html=True)
    else:
        st.info("Dataset not loaded — see setup instructions in the README.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='dishcovery-footer'>"
    "Dishcovery &nbsp;·&nbsp; Turn ingredients into meals"
    "&nbsp;·&nbsp; Built with Streamlit &amp; ❤️"
    "</div>",
    unsafe_allow_html=True,
)
