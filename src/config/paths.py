from pathlib import Path
from typing import NamedTuple


class Paths(NamedTuple):
    DATA_DIR = Path("data")
    OUTPUT_DIR = Path("data/output")
    BLEU_SCORES_DIR = Path("data/output/bleu_scores")