from typing import Union

import pandas as pd
from datasets import load_dataset
from loguru import logger
from torch.utils.data import DataLoader
from transformers import MarianMTModel, MarianTokenizer
from torchtext.data.metrics import bleu_score
from torchtext.data.metrics import bleu_score

def translate(
    text: Union[str, list[str]], model: MarianMTModel, tokenizer: MarianTokenizer
) -> list[str]:
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]


def get_dataloader() -> DataLoader:
    return DataLoader(
        dataset=load_dataset("wmt16", "de-en", split="test").with_format("torch")[
            "translation"
        ],
        batch_size=64,
        shuffle=True,
    )


def main(
    input_lang: str = "en",
    output_lang: str = "de",
    model_name: str = "Helsinki-NLP/opus-mt-en-de",
    debug: bool = False,
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
    tokenizer: MarianTokenizer = MarianTokenizer.from_pretrained(model_name)
    model: MarianMTModel = MarianMTModel.from_pretrained(model_name)

    # Getting the data
    dataloader: DataLoader = get_dataloader()

    if debug:
        sample: dict[str, list[str]] = next(iter(dataloader))
        model_results: list[str] = translate(
            sample.get(input_lang, []), model, tokenizer
        )
        pd.DataFrame(
            {
                input_lang: sample.get(input_lang, []),
                output_lang: sample.get(output_lang, []),
                "model": model_results,
            }
        ).to_csv("data/output/sample_output.csv")
    else:
        # Scoring translation using BLEU score
        reference_translations = []
        model_translations = []

        for batch in dataloader:
            input_texts = batch.get(input_lang, [])
            target_texts = batch.get(output_lang, [])

            model_results = translate(input_texts, model, tokenizer)

            reference_translations.extend(target_texts)
            model_translations.extend(model_results)

        bleu = bleu_score(model_translations, reference_translations)
        logger.info(f"BLEU score: {bleu}")
        


if __name__ == "__main__":
    main(debug=True)
