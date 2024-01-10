from typing import Optional

import typer
from loguru import logger
from torch.utils.data import DataLoader
from transformers import MarianMTModel, MarianTokenizer

from config.settings import Settings
from utils import create_bleu_score, create_sample, get_dataloader

app = typer.Typer()

settings = Settings()


@app.command()
def main(
    input_lang: Optional[str] = settings.input_lang,
    output_lang: Optional[str] = settings.output_lang,
    mt_model_name: Optional[str] = settings.mt_model_name,
    sample: Optional[bool] = settings.sample,
):
    """
    Main function for translation using a pre-trained model.

    Args:
        input_lang (str): Input language code.
        output_lang (str): Output language code.
        mt_model_name (str): Name of the pre-trained model.
        sample (bool): Flag indicating whether to run only sample.
    """
    # Getting a pre-trained model
    logger.debug(
        f"Settings: {input_lang} -> {output_lang}, model: {mt_model_name}, sample: {sample}"
    )
    tokenizer: MarianTokenizer = MarianTokenizer.from_pretrained(mt_model_name)
    model: MarianMTModel = MarianMTModel.from_pretrained(mt_model_name)
    logger.info(f"Model loaded: {mt_model_name}")

    # Getting the data
    dataloader: DataLoader = get_dataloader()
    logger.info("Data loaded")

    if sample:
        create_sample(dataloader, input_lang, output_lang, model, tokenizer)
    else:
        create_bleu_score(dataloader, input_lang, output_lang, model, tokenizer)


if __name__ == "__main__":
    app()
