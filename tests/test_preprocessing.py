"""
tests/test_preprocessing.py
============================
Unit tests for src.preprocessing.
"""

import pytest

from src.preprocessing import (
    clean_ingredient_list,
    normalise_ingredient,
    parse_ingredient_list,
    preprocess_dataframe,
)
import pandas as pd


class TestCleanIngredientList:
    def test_normal_ingredient_names(self):
        """Case 1: Normal ingredient names."""
        input_data = ["tomato", "garlic", "salt"]
        assert clean_ingredient_list(input_data) == ["tomato", "garlic", "salt"]

    def test_uppercase_and_lowercase_duplicates(self):
        """Case 2: Uppercase and lowercase duplicates."""
        input_data = ["Tomato", "tomato", "TOMATO"]
        assert clean_ingredient_list(input_data) == ["tomato"]

    def test_leading_and_trailing_whitespace(self):
        """Case 3: Leading and trailing whitespace."""
        input_data = ["  tomato  ", "\tgarlic\n", " salt "]
        assert clean_ingredient_list(input_data) == ["tomato", "garlic", "salt"]

    def test_empty_strings(self):
        """Case 4: Empty strings and whitespace-only strings."""
        input_data = ["tomato", "", "   ", "salt"]
        assert clean_ingredient_list(input_data) == ["tomato", "salt"]

    def test_empty_and_none_list(self):
        """Case 5: Empty ingredient list and None."""
        assert clean_ingredient_list([]) == []
        assert clean_ingredient_list(None) == []

    def test_duplicates_preserve_first_occurrence(self):
        """Case 6: Duplicate ingredients while preserving first occurrence."""
        input_data = ["tomatoes", "onion", "tomatoes", "salt", "onion"]
        assert clean_ingredient_list(input_data) == ["tomatoes", "onion", "salt"]

    def test_prompt_example(self):
        """Verify the exact example given in the user instructions."""
        input_data = [" Tomatoes ", "ONION", " tomatoes", "", " Salt "]
        expected = ["tomatoes", "onion", "salt"]
        assert clean_ingredient_list(input_data) == expected


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

    def test_handles_stringified_list(self):
        df = pd.DataFrame(
            {
                "recipe_id": ["r1"],
                "title": ["Cake"],
                "ingredients": ["[' Flour ', 'SUGAR', 'flour', '']"],
            }
        )
        result = preprocess_dataframe(df)
        assert result["ingredients_clean"].iloc[0] == ["flour", "sugar"]

    def test_handles_python_list(self):
        df = pd.DataFrame(
            {
                "recipe_id": ["r1"],
                "title": ["Cake"],
                "ingredients": [[" Flour ", "SUGAR", "flour", ""]],
            }
        )
        result = preprocess_dataframe(df)
        assert result["ingredients_clean"].iloc[0] == ["flour", "sugar"]

    def test_missing_column_raises(self):
        df = pd.DataFrame({"recipe_id": ["r1"], "title": ["Pasta"]})
        with pytest.raises(KeyError):
            preprocess_dataframe(df)
