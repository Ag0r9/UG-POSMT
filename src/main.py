from loguru import logger
from transformers import MarianMTModel, MarianTokenizer


def translate(text, model, tokenizer):
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]


def main():
    model_name = "Helsinki-NLP/opus-mt-en-de"
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    model = MarianMTModel.from_pretrained(model_name)

    test = translate("Please, have mercy", model, tokenizer)
    print(test)
    return


if __name__ == "__main__":
    main()
