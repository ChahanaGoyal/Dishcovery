# Dishcovery 🍽️

> **Turn what's in your fridge into tonight's dinner — no grocery run required.**

Dishcovery is a Streamlit-based recipe recommendation application. Enter the ingredients you have, choose a cooking-time limit, pick a recommendation method, and Dishcovery returns ranked recipes from the Food.com dataset. It was built as a college Machine Learning project.

---

## Table of Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [Dataset Setup](#dataset-setup)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Recommendation Methods](#recommendation-methods)
- [Running Tests](#running-tests)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Limitations](#limitations)

---

## Features

- **Ingredient input** — enter any ingredients you have, comma-separated.
- **Cooking-time filter** — limit results to recipes under 15, 30, or 60 minutes.
- **Two recommendation methods** — choose between Baseline (ingredient overlap) or Improved (TF-IDF cosine similarity).
- **Recipe cards** — each result shows the recipe name, score, cooking time, matched ingredients, and missing ingredients.
- **Recipe catalog preview** — browse a sample of the loaded dataset directly in the app.
- **Safe error handling** — clear messages when the dataset is missing, input is empty, or no recipes match.

---

## Technology Stack

| Component | Library / Tool |
|---|---|
| Web application | Streamlit ≥ 1.35 |
| Data handling | pandas ≥ 2.0 |
| Numerical computation | NumPy ≥ 2.0 |
| ML recommendation | scikit-learn ≥ 1.5 (TF-IDF, cosine similarity) |
| Testing | pytest ≥ 8.0 |
| Language | Python 3.10+ |

---

## Dataset Setup

Dishcovery uses the **Food.com Recipes and Interactions** dataset from Kaggle.

> **The dataset is not included in this repository** (it is too large and listed in `.gitignore`).

**Steps to set up the dataset:**

1. Go to the Kaggle dataset page:
   `https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions`

2. Download `RAW_recipes.csv`.

3. Place the file at:
   ```
   data/raw/RAW_recipes.csv
   ```

4. The application will load it automatically at startup.

If the file is missing, the app will display a clear setup message rather than crashing.

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ChahanaGoyal/Dishcovery.git
   cd Dishcovery
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Application

```bash
streamlit run app.py
```

The app opens in your default browser at `http://localhost:8501`.

**Typical workflow:**
1. Enter ingredients in the text area, separated by commas (e.g., `tomato, garlic, pasta`).
2. Select a cooking-time limit from the dropdown.
3. Choose a recommendation method.
4. Click **✨ Find Recipes**.
5. View ranked recipe cards below.

> **Note:** The app requires the dataset file `data/raw/RAW_recipes.csv` to display recommendations. See [Dataset Setup](#dataset-setup).

---

## Recommendation Methods

### Baseline — Ingredient Overlap

The baseline method counts how many of your entered ingredients appear in a recipe's ingredient list.

**Formula:**
```
Match % = (ingredients you have that the recipe needs / total recipe ingredients) × 100
```

- **Strength:** Fully explainable; the match percentage tells you exactly how close you are to having everything.
- **Limitation:** Treats all ingredients equally and does not account for ingredient importance or rarity.

### Improved — TF-IDF Cosine Similarity

The improved method uses **TF-IDF** (Term Frequency–Inverse Document Frequency) to represent each recipe's ingredient list as a weighted vector. Your ingredient query is transformed the same way, and recipes are ranked by **cosine similarity** to your query.

- **Strength:** Down-weights very common ingredients (e.g., salt, water) and up-weights distinctive ones, so rare or specific ingredients carry more influence.
- **Score label:** The score displayed is **Similarity %** (cosine similarity × 100), not a match percentage. A score of 80% means the ingredient profiles are highly similar, not that you have 80% of the ingredients.
- **Limitation:** The similarity score is less directly interpretable than a match percentage. Missing-ingredient information is still computed using the baseline overlap for transparency.

> **Important:** The similarity score and the match percentage are different metrics. Dishcovery labels them clearly to avoid confusion.

---

## Running Tests

```bash
python -m pytest tests/ -q
```

Expected output (dataset not required for unit tests):
```
123 passed, 2 skipped
```

The 2 skipped tests require the full Kaggle dataset file and are skipped automatically when it is absent.

### What the tests cover

- Recipe CSV loading and column validation (`test_data_loader.py`)
- Ingredient preprocessing: cleaning, deduplication, case normalisation (`test_preprocessing.py`)
- Baseline recommender: match score calculation, ranking, tie-breaking, edge cases (`test_recommender.py`)
- TF-IDF recommender: similarity range, ranking order, empty inputs (`test_recommender.py`)
- Evaluation metrics: Precision@K, Recall@K, F1@K; evaluation workflow (`test_evaluation.py`)
- Streamlit UI helpers: cooking-time filter, ingredient formatting, recipe card rendering, end-to-end recommendation flow (`test_ui_recipes.py`)

---

## Evaluation

`src/evaluation.py` provides a reproducible evaluation of the baseline recommender.

**Approach:**
1. Sample a subset of recipes from the dataset.
2. For each sampled recipe, take 50% of its ingredients as a simulated user query.
3. Run the recommender and check whether the original recipe appears in the top-5 results.
4. Report the retrieval rate and average ranking position.

**Run evaluation** (requires the dataset file):
```bash
python -m src.evaluation
```

**Limitations:**
- This is a self-retrieval evaluation (the recommender is tested on the same dataset it was trained on) and does not represent real user preferences.
- The baseline does not understand ingredient synonyms, quantities, or substitutions.
- Evaluation results may vary with different random seeds or sample sizes.

---

## Project Structure

```
Dishcovery/
├── app.py                    # Streamlit web application (entry point)
├── requirements.txt          # Python dependencies
├── README.md
├── .gitignore
├── data/
│   ├── raw/                  # Original datasets — place RAW_recipes.csv here (not committed)
│   └── processed/            # Transformed data (not committed)
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # CSV loading and column validation
│   ├── preprocessing.py      # Ingredient cleaning and normalisation
│   ├── recommender.py        # Baseline and TF-IDF recommendation functions
│   └── evaluation.py        # Evaluation metrics and workflow
└── tests/
    ├── test_data_loader.py
    ├── test_preprocessing.py
    ├── test_recommender.py
    ├── test_evaluation.py
    └── test_ui_recipes.py
```

---

## Limitations

- **No ingredient substitutions:** The recommender matches exact ingredient names and does not understand that "spring onion" and "scallion" are the same.
- **No quantity awareness:** A recipe requiring 2 kg of beef and 1 sprig of parsley are treated equally.
- **Dataset dependency:** The application is non-functional for recommendations without `data/raw/RAW_recipes.csv`.
- **Streamlit UI:** The app cannot be tested end-to-end in a headless CI environment; unit tests cover the individual functions instead.
- **English only:** The dataset and ingredient matching are designed for English ingredient names.
- **Self-retrieval evaluation:** The evaluation measures whether a recipe can be found when you enter some of its own ingredients — it is a consistency check, not a measure of recommendation quality for arbitrary user needs.
