# infer.py
#
# Interactive inference for homemade LLM.

from pathlib import Path

from tokenizer import Tokenizer
from tensor import TensorRuntime
from llm import LanguageModel


TOKENIZER_FILE = "model/tokenizer.json"
MODEL_FILE = "model/model-v0.3.json"

MAX_NEW_TOKENS = 100
GPU_MEMORY_SIZE = 8_000_000

TEMPERATURE = 0.8
TOP_K = 40
REPETITION_PENALTY = 1.15


def main():

    print()
    print("====================================")
    print(" Homemade LLM Interactive Inference v0.3")
    print("====================================")
    print()

    tokenizer = Tokenizer.load(
        TOKENIZER_FILE
    )

    print(
        "Vocabulary size:",
        tokenizer.vocab_size,
    )

    print()
    print("Creating virtual GPU runtime...")

    runtime = TensorRuntime(
        memory_size=GPU_MEMORY_SIZE,
    )

    print(
        "Virtual GPU memory:",
        f"{GPU_MEMORY_SIZE:,}",
        "float elements",
    )

    if not Path(MODEL_FILE).exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_FILE}. "
            "Run python train_corpus.py first."
        )

    print()
    print("Loading trained model checkpoint...")

    model = LanguageModel.load(
        MODEL_FILE,
        runtime=runtime,
    )

    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            "Tokenizer/model vocabulary mismatch: "
            f"{tokenizer.vocab_size} != {model.vocab_size}"
        )

    print(
        "Model loaded:",
        MODEL_FILE,
    )

    print()
    print("Enter Japanese text.")
    print("Type 'exit' to quit.")
    print()

    while True:

        prompt = input("You> ")

        if prompt.strip().lower() in (
            "exit",
            "quit",
        ):
            break

        if not prompt:
            continue

        input_ids = tokenizer.encode(
            prompt,
            add_bos=True,
        )

        output_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_id=tokenizer.eos_id,
            show_progress=True,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
        )

        generated_text = tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
        )

        print()
        print("LLM>", generated_text)
        print()


if __name__ == "__main__":
    main()
