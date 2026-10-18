"""
app.py – Dishcovery Streamlit Application
==========================================
Entry point for the Dishcovery web app.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

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

col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    find_clicked = st.button("✨ Find Recipes", use_container_width=True, type="primary")

# ── Result area ───────────────────────────────────────────────────────────────
if find_clicked:
    raw_ingredients = [i.strip() for i in ingredient_input.split(",") if i.strip()]

    if not raw_ingredients:
        st.warning("Please enter at least one ingredient before searching.")
    else:
        st.info(
            f"**{len(raw_ingredients)} ingredient(s) detected:** "
            + ", ".join(f"`{i}`" for i in raw_ingredients)
            + "\n\n"
            "🚧 The recommendation engine is coming soon! "
            "Dataset integration is in progress.",
            icon="🔍",
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
