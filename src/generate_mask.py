import spacy
from loguru import logger

try:
    en_core = spacy.load("en_core_web_sm")
except OSError:
    raise ModuleNotFoundError("core not found. Try `make lang`")

def extract_token_info(token):
    # Recursively extract information for a given token and its children
    token_info = {
        'text': token.text,
        'pos': token.pos_,
        'tag': token.tag_,
        'children': [extract_token_info(child) for child in token.children]
    }
    return token_info


def generate_mask(input: str):
    doc = en_core(input)
    tokens_info = []

    # Iterate through each token in the sentence
    for token in doc:
        # Extract relevant information for each token
        if token.dep_ == "ROOT":
            token_info = extract_token_info(token)
            # Append token information to the list
            tokens_info.append(token_info)
    logger.info(tokens_info)

if __name__ == "__main__":
    generate_mask("My name is Alice.")