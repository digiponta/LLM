# infer.py
#
# Interactive inference for homemade LLM.

from tokenizer import Tokenizer
from tensor import TensorRuntime
from llm import LanguageModel


TOKENIZER_FILE = "model/tokenizer.json"

D_MODEL = 64
NUM_LAYERS = 2
HIDDEN_DIM = 256

MAX_NEW_TOKENS = 100
GPU_MEMORY_SIZE = 8_000_000


def main():

    print()
    print("====================================")
    print(" Homemade LLM Interactive Inference")
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

    model = LanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
        causal=True,
        runtime=runtime,
        seed=42,
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
