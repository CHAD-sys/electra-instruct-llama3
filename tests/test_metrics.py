"""Tests for the eval metrics (pure functions, no model)."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.metrics import rouge_l, keyword_hit  # noqa: E402


def test_rouge_identical_is_one():
    assert rouge_l("a red led drops 2 volts", "a red led drops 2 volts") == 1.0


def test_rouge_disjoint_is_zero():
    assert rouge_l("apple banana", "resistor capacitor") == 0.0


def test_rouge_empty_inputs():
    assert rouge_l("", "anything") == 0.0
    assert rouge_l("anything", "") == 0.0


def test_rouge_partial_between_zero_and_one():
    score = rouge_l("the led drops two volts", "the led drops 2 volts")
    assert 0.0 < score < 1.0


def test_keyword_hit_all_present():
    assert keyword_hit("R is 120 ohm, Vf is 3.2", ["120", "ohm", "Vf"]) == 1.0


def test_keyword_hit_partial():
    assert keyword_hit("the answer is 120", ["120", "ohm"]) == 0.5


def test_keyword_hit_empty_is_one():
    assert keyword_hit("whatever", []) == 1.0


def test_keyword_hit_is_case_insensitive():
    assert keyword_hit("Brown band means 1%", ["brown", "1%"]) == 1.0
