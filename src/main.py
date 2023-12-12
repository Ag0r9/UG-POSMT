from loguru import logger
from typing import List, Union, Tuple
from transformers import MarianMTModel, MarianTokenizer
import pandas as pd


def get_data(data_file: str) -> Tuple[List, List]:
    data = pd.read_csv(data_file)
    de, eng = [], []
    for line in data.iloc[:, 1]:
        line = line.replace("{'de': ", " ")
        splitted = line.split(", 'en': ")
        de.append(splitted[0][1:])
        eng.append(splitted[1][:-1])
    return de, eng


def translate(
    text: Union[str, List], model: MarianMTModel, tokenizer: MarianTokenizer
) -> List:
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]


def main():
    # Getting a pre-trained model
    model_name = "Helsinki-NLP/opus-mt-en-de"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # loading data and dividing it to german and english
    data_file = "../data/data.csv"
    german_data, english_data = get_data(data_file)

    quantity = 5
    test = translate(english_data[:quantity], model, tokenizer)

    for x in range(quantity):
        print(
            english_data[x]
            + "\n"
            + german_data[x]
            + "\ntranslation:\n"
            + test[x]
            + "\n"
        )
    return


if __name__ == "__main__":
    main()
