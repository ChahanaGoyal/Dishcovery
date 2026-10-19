"""
evaluation.py
=============
Metrics for evaluating Dishcovery's recommendation quality.

Metrics implemented
-------------------
- Precision@K  : fraction of top-K recommendations that are relevant.
- Recall@K     : fraction of relevant items captured in the top-K.
- F1@K         : harmonic mean of Precision@K and Recall@K.

These functions operate on plain Python sets / lists and are independent
of any specific model, making them easy to unit-test.
"""

from __future__ import annotations


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
