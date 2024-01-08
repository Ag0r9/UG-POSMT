from config.paths import Paths

from statistics import mean

import pandas as pd
from datasets import load_dataset
from loguru import logger
from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

paths = Paths()

def get_dataloader() -> DataLoader:
    return DataLoader(
        dataset=load_dataset("wmt16", "de-en", split="test").with_format("torch")[
            "translation"
        ],
        batch_size=64,
        shuffle=True,
    )


def translate(
    text: str | list[str], model: MarianMTModel, tokenizer: MarianTokenizer
) -> list[str]:
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]


def create_sample(
    dataloader: DataLoader,
    input_lang: str,
    output_lang: str,
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
):
    sample: dict[str, list[str]] = next(iter(dataloader))
    model_results: list[str] = translate(sample.get(input_lang, []), model, tokenizer)
    pd.DataFrame(
        {
            input_lang: sample.get(input_lang, []),
            output_lang: sample.get(output_lang, []),
            "model": model_results,
        }
    ).to_csv(paths.OUTPUT_DIR / "sample_output.csv")


def create_bleu_score(
    dataloader: DataLoader,
    input_lang: str,
    output_lang: str,
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
):
    bleu_scores: list[float] = []

    for batch in tqdm(dataloader):
        input_texts: list[str] = batch.get(input_lang, [])
        target_texts: list[str] = batch.get(output_lang, [])

        model_results: list[str] = translate(input_texts, model, tokenizer)

        bleu_scores.append(bleu_score(model_results, target_texts))

    logger.info(f"BLEU score: {mean(bleu_scores)}")
