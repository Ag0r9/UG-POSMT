from typing import Any

import numpy as np
import spacy
from loguru import logger
from spacy.language import Language
from spacy.tokens import Doc, Token


def get_method(method_name: str) -> dict[str, bool]:
    methods: dict[str, tuple[bool]] = {
        "shallow": (False, False, True),
        "deep": (True, False, True),
        "only_children": (False, True, False),
        "only_parent": (True, False, False),
        "only_self": (False, False, True),
        "full": (True, True, True)
    }
    look_at_children, inherit_from_parent, look_at_yourself = methods[method_name]
    return {
        "look_at_children": look_at_children,
        "inherit_from_parent": inherit_from_parent,
        "look_at_yourself": look_at_yourself,
    }


class MaskGenerator:
    """
    Generates a mask based on the input text using spaCy language processing library.
    The mask represents the relationships between tokens in the text.

    Args:
        inherit_from_parent (bool, optional): Whether to inherit the mask value from the parent token.
            Defaults to False.
        look_at_children (bool, optional): Whether to include children tokens in the mask.
            Defaults to False.
        look_at_yourself (bool, optional): Whether to include the token itself in the mask.
            Defaults to True.
        input_lang (str, optional): The language of the input text. Defaults to "en".
    """
    def __init__(
        self,
        inherit_from_parent: bool = False,
        look_at_children: bool = False,
        look_at_yourself: bool = True,
        input_lang: str = "en",
    ) -> None:
        self.inherit_from_parent: bool = inherit_from_parent
        self.look_at_children: bool = look_at_children
        self.look_at_yourself: bool = look_at_yourself
        try:
            model_name = "en_core_web_sm"
            self.en_core: Language = spacy.load(model_name)
            logger.info(f"Loaded {model_name}")
        except OSError:
            raise ModuleNotFoundError("Language core not found. Try `make lang`")
        self.mask: np.ndarray

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

    def fill_mask(
        self, tokens_info: list[dict[str, Any]], parent_position: int | None = None
    ):
        for token in tokens_info:
            if parent_position:
                if self.inherit_from_parent:
                    self.mask[token.get("position")] = self.mask[parent_position]
                else:
                    self.mask[token.get("position"), parent_position] = 1
                    
            if self.look_at_yourself:
                self.mask[token.get("position"), token.get("position")] = 1

            if self.look_at_children:
                self.mask[
                    token.get("position"),
                    [child.get("position") for child in token.get("children", [])],
                ] = 1
            self.fill_mask(
                tokens_info=token.get("children", []),
                parent_position=token.get("position"),
            )

    def generate_mask(self, input: str):
        doc: Doc = self.en_core(input)
        tokens_info: list[dict[str, Any]] = []
        n: int = len(doc)
        self.mask = np.zeros((n, n))
        logger.info(f"Created empty mask with shape {self.mask.shape}")

        for token in doc:
            if token.dep_ == "ROOT":
                token_info: dict[str, Any] = self.extract_token_info(token=token)
                tokens_info.append(token_info)
        logger.debug(f"Extracted tokens info:\n{tokens_info}")

        self.fill_mask(tokens_info=tokens_info)
        logger.info(f"Mask filled for {input}")
        logger.debug(f"Created mask:\n{self.mask}")


if __name__ == "__main__":
    generator = MaskGenerator()
    generator.generate_mask(
        input="Apple is looking at buying U.K. startup for $1 billion."
    )
