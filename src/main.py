from loguru import logger
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    input_ids = tokenizer(
        "Studies have been shown that owning a dog is good for you", return_tensors="pt"
    ).input_ids  # Batch size 1
    decoder_input_ids = tokenizer(
        "Studies show that", return_tensors="pt"
    ).input_ids  # Batch size 1

    # forward pass
    outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    last_hidden_states = outputs.last_hidden_state
    logger.debug(last_hidden_states)
    return


if __name__ == "__main__":
    main()
