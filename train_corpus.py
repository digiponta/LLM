# train_corpus.py
#
# Train homemade LLM with:
#
#   data/general-ja.txt
#   data/data-nagato.txt
#
# v0.3:
#   - Trains Embedding, Transformer, Attention, FFN, LayerNorm, and lm_head.
#   - Uses a deliberately small first run to validate full backpropagation.
#   - Saves a checkpoint for infer.py.

from pathlib import Path

from tokenizer import Tokenizer
from tensor import TensorRuntime
from llm import LanguageModel
from train import Trainer, build_training_samples


GENERAL_FILE = "data/general-ja.txt"
NAGATO_FILE = "data/data-nagato.txt"

TOKENIZER_FILE = "model/tokenizer.json"
MODEL_FILE = "model/model-v0.3.json"

CONTEXT_LENGTH = 32

D_MODEL = 64
NUM_LAYERS = 2
HIDDEN_DIM = 256

# The virtual GPU stores float elements in a Python list.  The default
# TensorRuntime size (1,000,000 elements) is too small for a Japanese
# character vocabulary and a 64-token context because the LM head alone can
# require hundreds of thousands of temporary elements.  Use a larger runtime
# for corpus experiments while Trainer rewinds temporary allocations after
# every step.
GPU_MEMORY_SIZE = 8_000_000

# Full-model backward propagation is substantially more expensive than v0.2.
# Start small, confirm loss decreases and checkpoint inference works, then
# increase MAX_SAMPLES/EPOCHS.
EPOCHS = 1
LEARNING_RATE = 5e-4

# Full Wikipedia-derived corpora can produce millions of overlapping windows.
# v0.1 is a Python virtual-GPU experiment, so cap the number of samples while
# selecting them uniformly over the entire combined corpus.
MAX_SAMPLES = 200


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
    print(" Homemade LLM Corpus Training v0.3")
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

    total_possible_samples = max(
        1,
        len(token_ids) - CONTEXT_LENGTH,
    )

    samples = build_training_samples(
        token_ids,
        context_length=CONTEXT_LENGTH,
        max_samples=MAX_SAMPLES,
    )

    print(
        "Possible training samples:",
        f"{total_possible_samples:,}",
    )

    print(
        "Training samples used:",
        f"{len(samples):,}",
        f"(uniformly sampled, max={MAX_SAMPLES:,})",
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

    print()
    print("Creating model...")

    model = LanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
        causal=True,
        runtime=runtime,
        seed=42,
    )

    model.info()

    print()
    print("Persistent GPU memory after model creation:")
    runtime.memory.info()

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
    print("Saving model checkpoint...")

    model.save(
        MODEL_FILE
    )

    print(
        "Model saved:",
        MODEL_FILE,
    )

    print()
    print("NOTE:")
    print(
        "v0.3 performs full-model backward propagation and Adam updates "
        "across Embedding, Transformer, Attention, FFN, LayerNorm, and lm_head."
    )


if __name__ == "__main__":
    main()
