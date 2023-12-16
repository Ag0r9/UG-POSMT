import spacy
from loguru import logger
import numpy as np


class GenerateMask:
    def __init__(self) -> None:
        try:
            self.en_core = spacy.load("en_core_web_sm")
        except OSError:
            raise ModuleNotFoundError("core not found. Try `make lang`")
        self.mask: np.ndarray | None = None

    def extract_token_info(self, token):
        # Recursively extract information for a given token and its children
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

    def fill_mask(self, tokens_info, parent_position=None):
        for token in tokens_info:
            if parent_position:
                self.mask[token.get("position")] = self.mask[parent_position]
            self.mask[token.get("position"), token.get("position")] = 1
            self.fill_mask(
                tokens_info=token.get("children"), parent_position=token.get("position")
            )

    def generate_mask(self, input: str):
        doc = self.en_core(input)
        tokens_info = []
        n = len(doc)
        self.mask = np.zeros((n, n))

        # Iterate through each token in the sentence
        for token in doc:
            # Extract relevant information for each token
            if token.dep_ == "ROOT":
                token_info = self.extract_token_info(token=token)
                # Append token information to the list
                tokens_info.append(token_info)

        self.fill_mask(tokens_info=tokens_info)
        logger.info(f"Created mask:\n{self.mask}")


if __name__ == "__main__":
    generator = GenerateMask()
    generator.generate_mask(input="Apple is looking at buying U.K. startup for $1 billion.")

# TODO: patrzeć na rodzica tylko