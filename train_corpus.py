# train_corpus.py
#
# Train homemade LLM with:
#
#   data/general-ja.txt
#   data/data-nagato.txt
#
# Current v0.1 note:
# train.py currently calculates loss, but backward() and
# optimizer.step() are not connected yet.

from pathlib import Path

from tokenizer import Tokenizer
from llm import LanguageModel
from train import Trainer, build_training_samples


GENERAL_FILE = "data/general-ja.txt"
NAGATO_FILE = "data/data-nagato.txt"

TOKENIZER_FILE = "model/tokenizer.json"

CONTEXT_LENGTH = 64

D_MODEL = 64
NUM_LAYERS = 2
HIDDEN_DIM = 256

EPOCHS = 3
LEARNING_RATE = 1e-3


def load_text(filename):
    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {filename}"
        )

    return path.read_text(
        encoding="utf-8"
    )


def main():

    print()
    print("====================================")
    print(" Homemade LLM Corpus Training")
    print("====================================")
    print()

    print("Loading corpora...")

    general_text = load_text(GENERAL_FILE)
    nagato_text = load_text(NAGATO_FILE)

    print(
        "general-ja.txt:",
        f"{len(general_text):,}",
        "characters",
    )

    print(
        "data-nagato.txt:",
        f"{len(nagato_text):,}",
        "characters",
    )

    training_text = (
        general_text
        + "\n\n"
        + nagato_text
    )

    print(
        "Total:",
        f"{len(training_text):,}",
        "characters",
    )

    print()
    print("Building tokenizer...")

    tokenizer = Tokenizer()

    tokenizer.fit_texts(
        [
            general_text,
            nagato_text,
        ]
    )

    print(
        "Vocabulary size:",
        tokenizer.vocab_size,
    )

    Path("model").mkdir(
        exist_ok=True
    )

    tokenizer.save(
        TOKENIZER_FILE
    )

    print(
        "Tokenizer saved:",
        TOKENIZER_FILE,
    )

    print()
    print("Encoding corpus...")

    token_ids = tokenizer.encode(
        training_text,
        add_bos=True,
        add_eos=True,
    )

    print(
        "Token count:",
        f"{len(token_ids):,}",
    )

    print()
    print("Creating training samples...")

    samples = build_training_samples(
        token_ids,
        context_length=CONTEXT_LENGTH,
    )

    print(
        "Training samples:",
        f"{len(samples):,}",
    )

    print()
    print("Creating model...")

    model = LanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
        causal=True,
        seed=42,
    )

    model.info()

    trainer = Trainer(
        model=model,
        learning_rate=LEARNING_RATE,
    )

    print()
    print("Training...")
    print()

    history = trainer.train(
        samples,
        epochs=EPOCHS,
    )

    print()
    print("Training completed.")

    print(
        "Loss history:",
        history,
    )

    print()
    print("NOTE:")
    print(
        "Current v0.1 calculates loss, "
        "but model parameters are not yet "
        "updated because backward() and "
        "optimizer.step() are disabled."
    )


if __name__ == "__main__":
    main()
