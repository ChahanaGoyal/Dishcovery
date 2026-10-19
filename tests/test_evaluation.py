"""
tests/test_evaluation.py
=========================
Unit tests for src.evaluation metrics.
"""

import pytest

from src.evaluation import f1_at_k, precision_at_k, recall_at_k


RECOMMENDED = ["r1", "r2", "r3", "r4", "r5"]
RELEVANT = {"r1", "r3", "r5"}


class TestPrecisionAtK:
    def test_perfect(self):
        assert precision_at_k(["r1", "r3", "r5"], {"r1", "r3", "r5"}, k=3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["r2", "r4"], {"r1", "r3"}, k=2) == 0.0

    def test_partial(self):
        # 2 out of 3 are relevant
        result = precision_at_k(["r1", "r2", "r3"], {"r1", "r3"}, k=3)
        assert abs(result - 2 / 3) < 1e-9

    def test_k_larger_than_list(self):
        # Should not crash; numerator still uses k
        result = precision_at_k(["r1"], {"r1"}, k=5)
        assert result == 1 / 5

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            precision_at_k(RECOMMENDED, RELEVANT, k=0)


class TestRecallAtK:
    def test_perfect(self):
        assert recall_at_k(["r1", "r3", "r5"], {"r1", "r3", "r5"}, k=3) == 1.0

    def test_empty_relevant(self):
        assert recall_at_k(RECOMMENDED, set(), k=3) == 0.0

    def test_partial(self):
        # 2 out of 3 relevant found in top-3
        result = recall_at_k(["r1", "r2", "r3"], {"r1", "r3", "r5"}, k=3)
        assert abs(result - 2 / 3) < 1e-9

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            recall_at_k(RECOMMENDED, RELEVANT, k=-1)


class TestF1AtK:
    def test_perfect(self):
        assert f1_at_k(["r1", "r3", "r5"], {"r1", "r3", "r5"}, k=3) == 1.0

    def test_zero_when_no_hits(self):
        assert f1_at_k(["r2", "r4"], {"r1", "r3"}, k=2) == 0.0

    def test_harmonic_mean(self):
        p = precision_at_k(RECOMMENDED, RELEVANT, k=3)
        r = recall_at_k(RECOMMENDED, RELEVANT, k=3)
        expected = 2 * p * r / (p + r)
        assert abs(f1_at_k(RECOMMENDED, RELEVANT, k=3) - expected) < 1e-9
