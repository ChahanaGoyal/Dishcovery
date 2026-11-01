"""
evaluation.py
=============
Metrics and evaluation workflows for Dishcovery's recommendation system.

Metrics implemented
-------------------
- Precision@K  : fraction of top-K recommendations that are relevant.
- Recall@K     : fraction of relevant items captured in the top-K.
- F1@K         : harmonic mean of Precision@K and Recall@K.

Evaluation Workflows
--------------------
- evaluate_recommender: Baseline sanity evaluation that measures whether
  the recommender retrieves a recipe when given that recipe's own ingredients.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import pandas as pd

from src.recommender import _parse_and_clean_ingredients, rank_recipes

logger = logging.getLogger(__name__)


# ── Traditional Information Retrieval Metrics ─────────────────────────────────


def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Compute Precision@K.

    Parameters
    ----------
    recommended:
        Ordered list of recommended recipe IDs.
    relevant:
        Set of ground-truth relevant recipe IDs.
    k:
        Cut-off rank.

    Returns
    -------
    float
        Value in ``[0.0, 1.0]``.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    top_k = recommended[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / k


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Compute Recall@K.

    Parameters
    ----------
    recommended:
        Ordered list of recommended recipe IDs.
    relevant:
        Set of ground-truth relevant recipe IDs.
    k:
        Cut-off rank.

    Returns
    -------
    float
        Value in ``[0.0, 1.0]``. Returns ``0.0`` when *relevant* is empty.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / len(relevant)


def f1_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Compute F1@K (harmonic mean of Precision@K and Recall@K).

    Parameters
    ----------
    recommended:
        Ordered list of recommended recipe IDs.
    relevant:
        Set of ground-truth relevant recipe IDs.
    k:
        Cut-off rank.

    Returns
    -------
    float
        Value in ``[0.0, 1.0]``. Returns ``0.0`` when both P and R are zero.
    """
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ── Baseline Recommendation System Evaluation Workflow ────────────────────────


def _print_evaluation_report(
    summary: dict[str, int | float],
    sample_size: int,
    top_n: int,
    random_state: int,
) -> None:
    """Print an honest and human-readable terminal report."""
    print("=" * 72)
    print("Dishcovery — Baseline Recommender Evaluation")
    print("=" * 72)
    print("Configuration:")
    print(f"  - Target sample size : {sample_size}")
    print(f"  - Top-N cutoff       : {top_n}")
    print(f"  - Random seed        : {random_state}")
    print()
    print("Results:")
    print(f"  - Total test cases evaluated     : {summary['total_evaluated']}")
    print(f"  - Original recipes retrieved     : {summary['successful_retrievals']}")
    print(f"  - Original recipes not retrieved : {summary['unsuccessful_retrievals']}")
    print(f"  - Retrieval rate (@K={top_n})          : {summary['retrieval_rate']:.2f}%")
    print(f"  - Average match percentage       : {summary['average_match_percentage']:.2f}%")
    print()
    print("Interpretation & Limitations:")
    print("  * This evaluation feeds each recipe's complete ingredient list into")
    print("    the recommender as simulated user input.")
    print("  * It tests whether the system can retrieve a recipe when all of its")
    print("    required ingredients are supplied (a baseline sanity check).")
    print("  * It does NOT measure how well the system performs for real users")
    print("    who have only partial, missing, or approximate ingredients.")
    print("=" * 72)


def evaluate_recommender(
    df: pd.DataFrame,
    sample_size: int = 50,
    top_n: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> dict[str, int | float]:
    """Evaluate baseline recipe retrieval rate on a sample of recipes.

    Uses each sample recipe's own ingredients as simulated user input to check
    whether the recommender successfully retrieves the original recipe in its
    top-N recommendations.

    Important Interpretation Note:
    ------------------------------
    This sanity evaluation feeds complete ingredient lists from known recipes
    into the recommender. It verifies whether the matching and ranking logic
    retrieves a recipe when all of its required ingredients are provided.
    It does NOT measure performance for real-world scenarios where users supply
    only partial, missing, or approximate ingredients.

    Parameters
    ----------
    df:
        DataFrame of recipes.
    sample_size:
        Number of recipes with usable ingredients to sample (default: 50).
    top_n:
        Top-N recommendation cutoff (default: 5).
    random_state:
        Seed for reproducible sampling (default: 42).
    verbose:
        Whether to print a formatted terminal report (default: True).

    Returns
    -------
    dict[str, int | float]
        Dictionary with evaluation metrics.
    """
    if df is None or df.empty or sample_size <= 0:
        summary = {
            "total_evaluated": 0,
            "successful_retrievals": 0,
            "unsuccessful_retrievals": 0,
            "retrieval_rate": 0.0,
            "average_match_percentage": 0.0,
        }
        if verbose:
            _print_evaluation_report(summary, sample_size=sample_size, top_n=top_n, random_state=random_state)
        return summary

    id_col = "recipe_id" if "recipe_id" in df.columns else ("id" if "id" in df.columns else None)
    ing_col = (
        "ingredients_clean"
        if "ingredients_clean" in df.columns
        else ("ingredients" if "ingredients" in df.columns else None)
    )

    if ing_col is None:
        raise ValueError("DataFrame must contain an ingredients column.")

    ids = [str(x) for x in df[id_col].tolist()] if id_col else [str(i) for i in range(len(df))]
    raw_ings = df[ing_col].tolist()

    # Filter only recipes with usable ingredients
    valid_records: list[tuple[str, list[str]]] = []
    for rid, raw in zip(ids, raw_ings):
        cleaned = _parse_and_clean_ingredients(raw)
        if cleaned:
            valid_records.append((rid, cleaned))

    if not valid_records:
        summary = {
            "total_evaluated": 0,
            "successful_retrievals": 0,
            "unsuccessful_retrievals": 0,
            "retrieval_rate": 0.0,
            "average_match_percentage": 0.0,
        }
        if verbose:
            _print_evaluation_report(summary, sample_size=sample_size, top_n=top_n, random_state=random_state)
        return summary

    # Reproducible sampling using standard library random with fixed seed
    rng = random.Random(random_state)
    actual_sample_size = min(sample_size, len(valid_records))
    sampled = rng.sample(valid_records, actual_sample_size)

    successful = 0
    unsuccessful = 0
    retrieved_match_pcts: list[float] = []

    for original_id, ingredients in sampled:
        recommendations = rank_recipes(ingredients, df, top_n=top_n)
        rec_ids = [r.recipe_id for r in recommendations]

        if original_id in rec_ids:
            successful += 1
            for r in recommendations:
                if r.recipe_id == original_id:
                    retrieved_match_pcts.append(r.match_percentage)
                    break
        else:
            unsuccessful += 1

    total = len(sampled)
    retrieval_rate = round((successful / total) * 100.0, 2) if total > 0 else 0.0
    avg_match = (
        round(sum(retrieved_match_pcts) / len(retrieved_match_pcts), 2)
        if retrieved_match_pcts
        else 0.0
    )

    summary = {
        "total_evaluated": total,
        "successful_retrievals": successful,
        "unsuccessful_retrievals": unsuccessful,
        "retrieval_rate": retrieval_rate,
        "average_match_percentage": avg_match,
    }

    if verbose:
        _print_evaluation_report(summary, sample_size=sample_size, top_n=top_n, random_state=random_state)

    return summary


def run_baseline_evaluation(
    dataset_path: str | Path | None = None,
    nrows: int = 2000,
    sample_size: int = 50,
    top_n: int = 5,
    random_state: int = 42,
) -> dict[str, int | float]:
    """Load dataset slice and run baseline recommender evaluation.

    Parameters
    ----------
    dataset_path:
        Path to the recipe CSV file (defaults to data/raw/RAW_recipes.csv).
    nrows:
        Number of rows to load from CSV for fast evaluation (default: 2000).
    sample_size:
        Number of recipes to sample and evaluate (default: 50).
    top_n:
        Top-N recommendation cutoff (default: 5).
    random_state:
        Seed for reproducible sampling (default: 42).

    Returns
    -------
    dict[str, int | float]
        Evaluation summary metrics.
    """
    from src.data_loader import get_data_dir, load_csv

    if dataset_path is None:
        dataset_path = get_data_dir("raw") / "RAW_recipes.csv"

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    df = load_csv(path, nrows=nrows, usecols=["id", "name", "minutes", "ingredients"])
    return evaluate_recommender(
        df=df,
        sample_size=sample_size,
        top_n=top_n,
        random_state=random_state,
        verbose=True,
    )


if __name__ == "__main__":
    run_baseline_evaluation()
