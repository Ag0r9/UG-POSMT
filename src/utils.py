import json
from statistics import mean, median
from time import localtime, strftime

import pandas as pd
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

from config.paths import Paths

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
    paths.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
    custom_mask: str = "none",
):
    bleu_scores: list[float] = []
    for batch in tqdm(dataloader):
        input_texts: list[str] = batch.get(input_lang, [])
        target_texts: list[str] = batch.get(output_lang, [])

        model_results: list[str] = translate(input_texts, model, tokenizer)

        for model_text, target_text in zip(model_results, target_texts):
            bleu_scores.append(
                sentence_bleu(
                    references=[target_text.split()], hypothesis=model_text.split()
                )
            )

    filename: str = (
        f"{input_lang}_{output_lang}_{strftime('%Y%m%d_%H%M%S', localtime())}.json"
    )
    paths.BLEU_SCORES_DIR.mkdir(parents=True, exist_ok=True)
    with open(
        paths.BLEU_SCORES_DIR / filename,
        "w",
    ) as f:
        json.dump(
            {
                "model_name": model.name_or_path,
                "input_lang": input_lang,
                "output_lang": output_lang,
                "custom_mask": custom_mask,
                "BLEU score": {
                    "mean": mean(bleu_scores),
                    "median": median(bleu_scores),
                },
            },
            f,
        )
