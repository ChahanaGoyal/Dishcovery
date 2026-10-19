"""
tests/test_preprocessing.py
============================
Unit tests for src.preprocessing.
"""

import pytest

from src.preprocessing import (
    normalise_ingredient,
    parse_ingredient_list,
    preprocess_dataframe,
)
import pandas as pd


class TestNormaliseIngredient:
    def test_lowercases(self):
        assert normalise_ingredient("Garlic") == "garlic"

    def test_strips_punctuation(self):
        assert normalise_ingredient("olive oil!") == "olive oil"

    def test_collapses_whitespace(self):
        assert normalise_ingredient("  tomato   paste  ") == "tomato paste"

    def test_empty_string(self):
        assert normalise_ingredient("") == ""

    def test_mixed(self):
        result = normalise_ingredient("2 cups All-Purpose Flour!")
        assert "flour" in result
        assert result == result.lower()


class TestParseIngredientList:
    def test_basic_split(self):
        result = parse_ingredient_list("tomato, garlic, pasta")
        assert "tomato" in result
        assert "garlic" in result
        assert "pasta" in result
        assert len(result) == 3

    def test_strips_empty_parts(self):
        result = parse_ingredient_list("tomato,,garlic")
        assert len(result) == 2

    def test_custom_separator(self):
        result = parse_ingredient_list("tomato|garlic", sep="|")
        assert len(result) == 2


class TestPreprocessDataframe:
    def _make_df(self):
        return pd.DataFrame(
            {
                "recipe_id": ["r1", "r2"],
                "title": ["Pasta", "Salad"],
                "ingredients": ["tomato, garlic, pasta", "lettuce, tomato"],
            }
        )

    def test_adds_clean_column(self):
        df = self._make_df()
        result = preprocess_dataframe(df)
        assert "ingredients_clean" in result.columns

    def test_clean_column_is_list(self):
        df = self._make_df()
        result = preprocess_dataframe(df)
        assert isinstance(result["ingredients_clean"].iloc[0], list)

    def test_does_not_mutate_input(self):
        df = self._make_df()
        preprocess_dataframe(df)
        assert "ingredients_clean" not in df.columns

    def test_missing_column_raises(self):
        df = pd.DataFrame({"recipe_id": ["r1"], "title": ["Pasta"]})
        with pytest.raises(KeyError):
            preprocess_dataframe(df)
