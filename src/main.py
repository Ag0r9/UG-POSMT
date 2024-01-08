from typing import Optional

import typer
from torch.utils.data import DataLoader
from transformers import MarianMTModel, MarianTokenizer
from loguru import logger

from utils import create_bleu_score, create_sample, get_dataloader

app = typer.Typer()


@app.command()
def main(
    input_lang: Optional[str] = "en",
    output_lang: Optional[str] = "de",
    model_name: Optional[str] = "Helsinki-NLP/opus-mt-en-de",
    debug: Optional[bool] = True,
):
    """
    Main function for translation using a pre-trained model.

    Args:
        input_lang (str): Input language code (default is "en").
        output_lang (str): Output language code (default is "de").
        model_name (str): Name of the pre-trained model (default is "Helsinki-NLP/opus-mt-en-de").
        debug (bool): Flag indicating whether to run in debug mode (default is False).
    """
    # Getting a pre-trained model
    logger.info(f"Settings: {input_lang} -> {output_lang}, {model_name}, debug: {debug}")
    tokenizer: MarianTokenizer = MarianTokenizer.from_pretrained(model_name)
    model: MarianMTModel = MarianMTModel.from_pretrained(model_name)
    logger.info(f"Model loaded: {model_name}")

    # Getting the data
    dataloader: DataLoader = get_dataloader()
    logger.info("Data loaded")

    if debug:
        create_sample(dataloader, input_lang, output_lang, model, tokenizer)
    else:
        create_bleu_score(dataloader, input_lang, output_lang, model, tokenizer)


if __name__ == "__main__":
    main()
