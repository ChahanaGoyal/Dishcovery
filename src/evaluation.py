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
- evaluate_recommender: Evaluates recommendation performance by sampling recipes,
  constructing simulated partial ingredient inputs, and checking retrieval in top-N.
- run_baseline_evaluation: Convenience runner that loads a sample of the raw dataset
  and produces a formatted terminal evaluation report.
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
    summary: dict[str, int | float | None],
    sample_size: int,
    top_n: int,
    random_state: int,
    partial_ratio: float,
) -> None:
    """Print an honest and human-readable terminal report."""
    print("=" * 72)
    print("Dishcovery — Baseline Recommender Evaluation")
    print("=" * 72)
    print("Configuration:")
    print(f"  - Target sample size       : {sample_size}")
    print(f"  - Top-N retrieval cutoff   : {top_n}")
    print(f"  - Random seed              : {random_state}")
    print(f"  - Partial ingredient ratio : {int(partial_ratio * 100)}% of recipe ingredients")
    print()
    print("Results:")
    print(f"  - Valid test cases evaluated     : {summary['valid_test_cases']}")
    print(f"  - Original recipes retrieved     : {summary['successful_retrievals']}")
    print(f"  - Original recipes not retrieved : {summary['unsuccessful_retrievals']}")
    print(f"  - Retrieval rate (@K={top_n})          : {summary['retrieval_rate']:.2f}%")
    if summary["average_ranking_position"] is not None:
        print(f"  - Average ranking position       : {summary['average_ranking_position']:.2f}")
    else:
        print("  - Average ranking position       : N/A (no retrievals)")
    print(f"  - Average match percentage       : {summary['average_match_percentage']:.2f}%")
    print()
    print("Interpretation & Limitations:")
    print("  * This evaluation simulates partial ingredient searches by selecting a")
    print(f"    subset ({int(partial_ratio * 100)}%) of each test recipe's ingredients.")
    print("  * Retrieval rate measures whether the target recipe appears in top-N")
    print("    recommendations when a user enters only some of its ingredients.")
    print("  * It does NOT guarantee that the retrieved dishes match all user preferences,")
    print("    dietary restrictions, or culinary synergies.")
    print("  * The baseline recommender uses direct ingredient overlap and does not")
    print("    understand ingredient synonyms, substitutions, or quantities.")
    print("=" * 72)


def evaluate_recommender(
    df: pd.DataFrame,
    sample_size: int = 50,
    top_n: int = 5,
    random_state: int = 42,
    partial_ratio: float = 0.5,
    verbose: bool = True,
) -> dict[str, int | float | None]:
    """Evaluate baseline recipe retrieval rate on a sample of recipes.

    Uses a reproducible subset of each recipe's ingredients to simulate realistic
    partial ingredient queries, and checks whether the recommender successfully
    retrieves the original recipe in its top-N recommendations.

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
    partial_ratio:
        Fraction of ingredients to select for each simulated user input in (0.0, 1.0] (default: 0.5).
        If 1.0, all ingredients are used (full-match sanity test).
    verbose:
        Whether to print a formatted terminal report (default: True).

    Returns
    -------
    dict[str, int | float | None]
        Dictionary with evaluation metrics:
        - valid_test_cases / total_evaluated
        - successful_retrievals
        - unsuccessful_retrievals
        - retrieval_rate
        - average_ranking_position / average_rank
        - average_match_percentage
        - partial_ratio
        - top_n_cutoff
    """
    if not isinstance(partial_ratio, (int, float)) or not (0.0 < partial_ratio <= 1.0):
        raise ValueError(f"partial_ratio must be a number in (0.0, 1.0], got {partial_ratio!r}")

    if df is None or df.empty or sample_size <= 0:
        summary: dict[str, int | float | None] = {
            "valid_test_cases": 0,
            "total_evaluated": 0,
            "successful_retrievals": 0,
            "unsuccessful_retrievals": 0,
            "retrieval_rate": 0.0,
            "average_ranking_position": None,
            "average_rank": None,
            "average_match_percentage": 0.0,
            "partial_ratio": partial_ratio,
            "top_n_cutoff": top_n,
        }
        if verbose:
            _print_evaluation_report(summary, sample_size=sample_size, top_n=top_n, random_state=random_state, partial_ratio=partial_ratio)
        return summary

    id_col = "recipe_id" if "recipe_id" in df.columns else ("id" if "id" in df.columns else None)
    ing_col = (
        "ingredients_clean"
        if "ingredients_clean" in df.columns
        else ("ingredients" if "ingredients" in df.columns else None)
    )

    if ing_col is None:
        raise ValueError("DataFrame must contain an ingredients column ('ingredients_clean' or 'ingredients').")

    ids = [str(x) for x in df[id_col].tolist()] if id_col else [str(i) for i in range(len(df))]
    raw_ings = df[ing_col].tolist()

    # Filter only recipes with usable ingredients (at least 1 ingredient)
    valid_records: list[tuple[str, list[str]]] = []
    for rid, raw in zip(ids, raw_ings):
        cleaned = _parse_and_clean_ingredients(raw)
        if cleaned:
            valid_records.append((rid, cleaned))

    if not valid_records:
        summary = {
            "valid_test_cases": 0,
            "total_evaluated": 0,
            "successful_retrievals": 0,
            "unsuccessful_retrievals": 0,
            "retrieval_rate": 0.0,
            "average_ranking_position": None,
            "average_rank": None,
            "average_match_percentage": 0.0,
            "partial_ratio": partial_ratio,
            "top_n_cutoff": top_n,
        }
        if verbose:
            _print_evaluation_report(summary, sample_size=sample_size, top_n=top_n, random_state=random_state, partial_ratio=partial_ratio)
        return summary

    # Reproducible sampling using standard library random with fixed seed
    rng = random.Random(random_state)
    actual_sample_size = min(sample_size, len(valid_records))
    sampled = rng.sample(valid_records, actual_sample_size)

    successful = 0
    unsuccessful = 0
    ranks: list[int] = []
    retrieved_match_pcts: list[float] = []

    for original_id, ingredients in sampled:
        # Construct partial ingredient input
        if partial_ratio < 1.0:
            k = max(1, int(len(ingredients) * partial_ratio))
            simulated_input = rng.sample(ingredients, k)
        else:
            simulated_input = ingredients

        recommendations = rank_recipes(simulated_input, df, top_n=top_n)
        rec_ids = [r.recipe_id for r in recommendations]

        if original_id in rec_ids:
            successful += 1
            rank_pos = rec_ids.index(original_id) + 1
            ranks.append(rank_pos)
            retrieved_match_pcts.append(recommendations[rank_pos - 1].match_percentage)
        else:
            unsuccessful += 1

    total = len(sampled)
    retrieval_rate = round((successful / total) * 100.0, 2) if total > 0 else 0.0
    avg_rank = round(sum(ranks) / len(ranks), 2) if ranks else None
    avg_match = (
        round(sum(retrieved_match_pcts) / len(retrieved_match_pcts), 2)
        if retrieved_match_pcts
        else 0.0
    )

    summary = {
        "valid_test_cases": total,
        "total_evaluated": total,
        "successful_retrievals": successful,
        "unsuccessful_retrievals": unsuccessful,
        "retrieval_rate": retrieval_rate,
        "average_ranking_position": avg_rank,
        "average_rank": avg_rank,
        "average_match_percentage": avg_match,
        "partial_ratio": partial_ratio,
        "top_n_cutoff": top_n,
    }

    if verbose:
        _print_evaluation_report(summary, sample_size=sample_size, top_n=top_n, random_state=random_state, partial_ratio=partial_ratio)

    return summary


def run_baseline_evaluation(
    dataset_path: str | Path | None = None,
    nrows: int = 2000,
    sample_size: int = 50,
    top_n: int = 5,
    random_state: int = 42,
    partial_ratio: float = 0.5,
) -> dict[str, int | float | None]:
    """Load dataset slice and run baseline recommender evaluation on partial inputs.

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
    partial_ratio:
        Fraction of ingredients to select for simulated user input (default: 0.5).

    Returns
    -------
    dict[str, int | float | None]
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
        partial_ratio=partial_ratio,
        verbose=True,
    )


if __name__ == "__main__":
    run_baseline_evaluation()
