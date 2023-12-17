import numpy as np
import pytest
from spacy.tokens import Doc
from src.MaskGenerator import MaskGenerator


@pytest.fixture
def mask_generator():
    return MaskGenerator()


@pytest.fixture
def tokens_info():
    return [
        {
            "text": "is",
            "pos": "AUX",
            "tag": "VBZ",
            "position": 1,
            "children": [
                {
                    "text": "This",
                    "pos": "PRON",
                    "tag": "DT",
                    "position": 0,
                    "children": [],
                },
                {
                    "text": "test",
                    "pos": "NOUN",
                    "tag": "NN",
                    "position": 3,
                    "children": [
                        {
                            "text": "a",
                            "pos": "DET",
                            "tag": "DT",
                            "position": 2,
                            "children": [],
                        }
                    ],
                },
            ],
        }
    ]


def test_extract_token_info(mask_generator):
    doc = Doc(mask_generator.en_core.vocab, words=["This", "is", "a", "test"])
    token = doc[0]
    token_info = mask_generator.extract_token_info(token)
    assert token_info == {
        "text": "This",
        "pos": "",
        "tag": "",
        "position": 0,
        "children": [],
    }


def test_generate_mask(mask_generator):
    input_text = "This is a test"
    expected_mask = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1]])
    mask_generator.generate_mask(input_text)
    assert np.array_equal(mask_generator.mask, expected_mask)
