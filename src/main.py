from typing import List, Tuple, Union

import pandas as pd
from datasets import load_dataset
from loguru import logger
from torch.utils.data import DataLoader
from transformers import MarianMTModel, MarianTokenizer


def translate(
    text: Union[str, List], model: MarianMTModel, tokenizer: MarianTokenizer
) -> List:
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]


def get_dataloader() -> DataLoader:
    dataset = load_dataset("wmt16", "de-en", split="test")
    dataset_pt = dataset.with_format("torch")["translation"]
    return DataLoader(dataset_pt, batch_size=64, shuffle=True)


def main(debug: bool = False):
    # Getting a pre-trained model
    model_name = "Helsinki-NLP/opus-mt-en-de"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Getting the data
    dataloader = get_dataloader()

    if debug:
        sample = next(iter(dataloader))
        model_results = translate(sample.get("en"), model, tokenizer)
        pd.DataFrame(
            {"en": sample.get("en"), "de": sample.get("de"), "model": model_results}
        ).to_csv("data/results.csv")
    else:
        logger.debug(
            "Trzeba zainplementować mierzenie jakości tłumaczenia dla całego datasetu"
        )


if __name__ == "__main__":
    main(debug=True)

# Nie jest nam potrzebne
# class TranslationDataset(Dataset):
#     def __init__(self, data_list):
#         self.data_list = data_list

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, index):
#         return self.data_list[index]
