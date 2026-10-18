# Dishcovery 🍽️

> **Turn what's in your fridge into tonight's dinner — no grocery run required.**

Dishcovery is an ML-based recipe recommendation app that suggests recipes based on the ingredients you already have at home.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the App](#running-the-app)
- [Running Tests](#running-tests)
- [Contributing](#contributing)

---

## Project Structure

```
Dishcovery/
├── app.py              # Streamlit web application (entry point)
├── requirements.txt    # Python dependencies
├── README.md
├── .gitignore
├── data/
│   ├── raw/            # Original, unmodified datasets (not committed)
│   └── processed/      # Cleaned / transformed data (not committed)
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Dataset loading & validation
│   ├── preprocessing.py    # Text cleaning & ingredient normalisation
│   ├── recommender.py      # Ingredient-based recommendation engine
│   └── evaluation.py       # Metrics: Precision@K, Recall@K, F1@K
├── tests/              # Pytest test suite
└── assets/             # Images, icons, and other static files
```

---

## Setup

### Prerequisites

- Python **3.10+**
- `pip` (or a compatible package manager)

### 1 — Clone the repository

```bash
git clone https://github.com/ChahanaGoyal/Dishcovery.git
cd Dishcovery
```

### 2 — Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Contributing

This project uses three branches:

| Branch    | Owner    | Purpose                          |
|-----------|----------|----------------------------------|
| `main`    | shared   | Stable, reviewed code only       |
| `chahana` | Chahana  | Feature development              |
| `anshika` | Anshika  | Feature development              |

**Workflow:**
1. Always work on your assigned branch.
2. Keep commits focused and descriptive.
3. Open a pull request to `main` when a feature is ready for review.
4. Never force-push or rebase shared history.

---

*Built with [Streamlit](https://streamlit.io), [scikit-learn](https://scikit-learn.org), [Pandas](https://pandas.pydata.org) and ❤️.*
