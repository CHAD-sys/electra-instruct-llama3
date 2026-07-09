"""Tests for the data-building pipeline. No GPU / model needed."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.build_dataset import (  # noqa: E402
    parse_doc,
    make_pairs,
    make_negative_pairs,
    paraphrase,
)

SAMPLE_DOC = """LM7805 linear voltage regulator
Output voltage: 5 V (fixed)
Output current: up to 1 A

The LM7805 is a three-terminal positive fixed regulator.
"""


def test_parse_doc_extracts_name_fields_notes():
    name, fields, notes = parse_doc(SAMPLE_DOC)
    assert name == "LM7805 linear voltage regulator"
    assert fields["output voltage"] == "5 V (fixed)"
    assert fields["output current"] == "up to 1 A"
    assert any("three-terminal" in n for n in notes)


def test_parse_doc_empty():
    name, fields, notes = parse_doc("   \n\n")
    assert name is None and fields == {} and notes == []


def test_make_pairs_has_one_qa_per_field():
    name, fields, notes = parse_doc(SAMPLE_DOC)
    pairs = make_pairs(name, fields, notes)
    # 2 fields + 1 explain pair from the notes
    assert len(pairs) == 3
    for p in pairs:
        assert p["instruction"] and p["output"]
        assert set(p.keys()) == {"instruction", "input", "output"}


def test_negatives_are_honest_and_absent_from_fields():
    name, fields, _ = parse_doc(SAMPLE_DOC)
    negs = make_negative_pairs(name, fields, n=2)
    assert 1 <= len(negs) <= 2
    for p in negs:
        # the asked-about field must not be one the doc actually lists
        assert "isn't specified" in p["output"] or "not" in p["output"].lower()


def test_paraphrase_preserves_output_and_adds_variants():
    base = [{"instruction": "What is the output voltage?", "input": "",
             "output": "5 V"}]
    variants = paraphrase(base, n_variants=2)
    assert len(variants) == 2
    for v in variants:
        assert v["output"] == "5 V"          # answer unchanged
        assert v["instruction"] != base[0]["instruction"]  # phrasing changed


def test_paraphrase_off_when_zero():
    base = [{"instruction": "x", "input": "", "output": "y"}]
    assert paraphrase(base, n_variants=0) == []
