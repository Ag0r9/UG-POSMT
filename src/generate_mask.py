from typing import Any

import numpy as np
import spacy
from loguru import logger
from spacy.language import Language
from spacy.tokens import Doc, Token


class MaskGenerator:
    """
    Generates a mask based on the input text using spaCy library.

    Args:
        shallow (bool, optional): Specifies whether to use shallow masking. Defaults to True.

    Attributes:
        shallow (bool): Specifies whether to use shallow masking.
        en_core (Language): The spaCy language model.
        mask (np.ndarray | None): The generated mask.

    Methods:
        extract_token_info: Extracts information about a token.
        fill_mask: Fills the mask based on the token information.
        generate_mask: Generates the mask based on the input text.
    """

    def __init__(self, shallow: bool = True) -> None:
        self.shallow: bool = shallow
        try:
            self.en_core: Language = spacy.load("en_core_web_sm")
        except OSError:
            raise ModuleNotFoundError("core not found. Try `make lang`")
        self.mask: np.ndarray | None = None

    def extract_token_info(self, token: Token) -> dict[str, Any]:
        token_info = {
            "text": token.text,
            "pos": token.pos_,
            "tag": token.tag_,
            "position": token.i,
            "children": [
                self.extract_token_info(token=child) for child in token.children
            ],
        }
        return token_info

    def fill_mask(self, tokens_info: dict[str, Any], parent_position=None):
        for token in tokens_info:
            if parent_position:
                if self.shallow:
                    self.mask[token.get("position"), parent_position] = 1
                else:
                    self.mask[token.get("position")] = self.mask[parent_position]
            self.mask[token.get("position"), token.get("position")] = 1
            self.fill_mask(
                tokens_info=token.get("children"), parent_position=token.get("position")
            )

    def generate_mask(self, input: str):
        doc: Doc = self.en_core(input)
        logger.debug(type(doc))
        tokens_info: list[dict[str, Any]] = []
        n: int = len(doc)
        self.mask = np.zeros((n, n))

        for token in doc:
            if token.dep_ == "ROOT":
                token_info = self.extract_token_info(token=token)
                tokens_info.append(token_info)

        self.fill_mask(tokens_info=tokens_info)
        logger.info(f"Created mask:\n{self.mask}")


if __name__ == "__main__":
    generator = MaskGenerator()
    generator.generate_mask(
        input="Apple is looking at buying U.K. startup for $1 billion."
    )
