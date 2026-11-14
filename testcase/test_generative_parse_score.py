"""Test cases for dna_factory/rewards/generative.py::_parse_score"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from dna_factory.rewards.generative import _parse_score


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Good answer.\nScore: 8", 1.0),
        ("Weak answer.\nScore: 3", 0.0),
        ("Score: 3 would be too harsh, it is mostly right.\nScore: 8", 1.0),
        ("Score: 9 is too generous given the error.\nScore: 4", 0.0),
        ("<think>maybe Score: 2</think>\nScore: 9", 1.0),
        ("<think>Score: 9?</think>Score: 1", 0.0),
        ("no score here", None),
        ("Score: 11", None),
    ],
)
def test_parse_score_uses_final_score(text, expected):
    assert _parse_score(text) == expected
