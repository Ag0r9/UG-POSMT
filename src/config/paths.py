from pathlib import Path
from typing import NamedTuple


class Paths(NamedTuple):
    DATA_DIR: Path = Path("data")
    OUTPUT_DIR: Path = DATA_DIR / "output"
    BLEU_SCORES_DIR: Path = OUTPUT_DIR / "bleu_scores"

    CONFIG_DIR: Path = Path("src/config")
