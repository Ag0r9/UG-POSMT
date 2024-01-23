from pathlib import Path
from typing import NamedTuple


class Paths(NamedTuple):
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    OUTPUT_DIR: Path = DATA_DIR / "output"
    BLEU_SCORES_DIR: Path = OUTPUT_DIR / "bleu_scores"

    CONFIG_DIR: Path = BASE_DIR / "src/config"
