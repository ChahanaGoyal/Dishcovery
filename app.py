"""
app.py – Dishcovery Streamlit Application
==========================================
Entry point for the Dishcovery web app.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import get_data_dir, load_csv, validate_foodcom_columns

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
        /* Gradient header */
        .hero-title {
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #FF6B35 0%, #F7931E 50%, #FFD700 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            margin-bottom: 0.2rem;
        }
        .hero-tagline {
            font-size: 1.2rem;
            color: #6B7280;
            text-align: center;
            margin-bottom: 2.5rem;
        }
        .section-header {
            font-size: 1.1rem;
            font-weight: 600;
            color: #374151;
            margin-bottom: 0.5rem;
        }
        .info-card {
            background: #FFF7ED;
            border-left: 4px solid #F7931E;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin-top: 2rem;
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

st.divider()

# ── Ingredient input ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🥕 What ingredients do you have?</div>', unsafe_allow_html=True)

ingredient_input = st.text_area(
    label="Ingredients",
    placeholder="e.g.  tomato, garlic, pasta, olive oil, basil",
    height=120,
    label_visibility="collapsed",
    help="Enter your available ingredients separated by commas.",
)

# ── Cooking-time filter ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">⏱️ Maximum cooking time</div>', unsafe_allow_html=True)

cooking_time = st.selectbox(
    label="Cooking time",
    options=["Any cooking time", "Up to 15 minutes", "Up to 30 minutes", "Up to 60 minutes"],
    index=0,
    label_visibility="collapsed",
    help="Filter recipes by how long they take to prepare.",
)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    find_clicked = st.button("✨ Find Recipes", use_container_width=True, type="primary")


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


# ── Result area ───────────────────────────────────────────────────────────────
st.divider()
st.markdown('<div class="section-header">📋 Recipe Recommendations</div>', unsafe_allow_html=True)

if find_clicked:
    cleaned_ingredients = parse_ingredients(ingredient_input)

    if not cleaned_ingredients:
        st.warning("Please enter at least one ingredient before searching.")
    else:
        time_label = cooking_time if cooking_time != "Any cooking time" else "any duration"
        st.info(
            f"**Cleaned ingredients ({len(cleaned_ingredients)}):** "
            + ", ".join(f"`{i}`" for i in cleaned_ingredients)
            + f"\n\n**Cooking time filter:** {time_label}\n\n"
            "🚧 The recommendation engine is coming soon! "
            "Recipe recommendations based on your ingredients will be added in a later development step.",
            icon="🔍",
        )
else:
    st.markdown(
        "<p style='color:#9CA3AF; font-style:italic;'>"
        "Enter your ingredients above and click <strong>Find Recipes</strong> to get started. "
        "Recipe recommendations will be added in a later development step."
        "</p>",
        unsafe_allow_html=True,
    )

# ── Recipe Catalog Sample ─────────────────────────────────────────────────────
st.divider()
st.markdown('<div class="section-header">🍲 Recipe Catalog Preview</div>', unsafe_allow_html=True)

recipes_df, load_err = load_recipe_catalog()

if load_err:
    st.warning(load_err)
elif recipes_df is not None and not recipes_df.empty:
    filtered_recipes = filter_by_cooking_time(recipes_df, cooking_time)

    st.caption(
        f"Showing up to 5 sample recipes from the dataset matching: **{cooking_time}** "
        f"({len(filtered_recipes)} matching in catalog). "
        "These are sample entries to verify data loading; personalized recommendation ranking will be added soon."
    )

    if filtered_recipes.empty:
        st.info("No recipes found within this cooking-time limit.")
    else:
        sample_to_show = filtered_recipes.head(5)
        for _, row in sample_to_show.iterrows():
            name = row.get("name")
            recipe_title = str(name).title() if pd.notna(name) and str(name).strip() else "Untitled Recipe"

            raw_min = row.get("minutes")
            if pd.notna(raw_min):
                try:
                    min_val = float(raw_min)
                    time_str = f"{int(min_val)} mins" if min_val.is_integer() else f"{min_val} mins"
                except (ValueError, TypeError):
                    time_str = f"{raw_min} mins"
            else:
                time_str = "Time not specified"

            ing_str = format_recipe_ingredients(row.get("ingredients"))

            with st.container():
                st.markdown(f"**🍽️ {recipe_title}** &nbsp;·&nbsp; ⏱️ *{time_str}*")
                st.markdown(
                    f"<p style='color:#4B5563; font-size:0.9rem; margin-top:-0.3rem; margin-bottom:0.75rem;'>"
                    f"<strong>Ingredients:</strong> {ing_str}"
                    f"</p>",
                    unsafe_allow_html=True,
                )

# ── Footer info card ──────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="info-card">
        <strong>🛠️ Project status</strong><br>
        Dishcovery is under active development.
        The ML recommendation engine will be connected once the dataset pipeline is finalised.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<br><p style='text-align:center; color:#9CA3AF; font-size:0.85rem;'>"
    "Dishcovery · Built with Streamlit &amp; ❤️"
    "</p>",
    unsafe_allow_html=True,
)
