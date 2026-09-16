# demo.py
#
# Demo program for the homemade LLM.
#
# Flow:
#
#   Text
#    |
#    v
#   Tokenizer
#    |
#    v
#   Token IDs
#    |
#    v
#   LanguageModel
#    |
#    v
#   Logits
#    |
#    v
#   Next Token
#    |
#    v
#   Generated Text
#
# Note:
#   The model is not trained yet.
#   Therefore generated text is only a functional demo.

from tokenizer import Tokenizer
from llm import LanguageModel


def main():

    print()
    print("================================")
    print("Homemade LLM Demo")
    print("================================")
    print()

    # --------------------------------------------------------
    # Small demonstration corpus
    #
    # This is used only to build the tokenizer vocabulary.
    # It is NOT training the model.
    # --------------------------------------------------------

    corpus = (
        "hello world "
        "hello llm "
        "hello gpu "
        "python transformer "
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = Tokenizer()

    tokenizer.fit(
        corpus
    )

    print("Tokenizer")
    print("---------")
    print(
        "Vocabulary size:",
        tokenizer.vocab_size
    )
    print()

    tokenizer.dump_vocab()

    print()

    # --------------------------------------------------------
    # Language Model
    # --------------------------------------------------------

    model = LanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=8,
        num_layers=1,
        hidden_dim=32,
        causal=True,
        seed=42
    )

    print("Model")
    print("-----")

    model.info()

    print()

    print(
        "Trainable parameter tensors:",
        len(
            model.parameters()
        )
    )

    print()

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = "hello"

    print("Prompt")
    print("------")
    print(
        repr(prompt)
    )

    print()

    # --------------------------------------------------------
    # Encode
    # --------------------------------------------------------

    token_ids = tokenizer.encode(
        prompt,
        add_bos=True
    )

    print("Token IDs")
    print("---------")
    print(
        token_ids
    )

    print()

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    logits = model(
        token_ids
    )

    print("Forward Pass")
    print("------------")

    print(
        "Logits shape:",
        logits.shape
    )

    print()

    # --------------------------------------------------------
    # Next Token Prediction
    # --------------------------------------------------------

    next_token_id = model.next_token(
        token_ids
    )

    print("Next Token")
    print("----------")

    print(
        "Predicted token ID:",
        next_token_id
    )

    next_token_text = tokenizer.decode(
        [
            next_token_id
        ],
        skip_special_tokens=False
    )

    print(
        "Predicted token:",
        repr(
            next_token_text
        )
    )

    print()

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    probabilities = (
        model.last_probabilities(
            token_ids
        )
    )

    print("Top Predictions")
    print("---------------")

    indexed = list(
        enumerate(
            probabilities
        )
    )

    indexed.sort(
        key=lambda item: item[1],
        reverse=True
    )

    top_k = min(
        5,
        len(indexed)
    )

    for rank in range(
        top_k
    ):

        token_id, probability = (
            indexed[
                rank
            ]
        )

        token_text = tokenizer.decode(
            [
                token_id
            ],
            skip_special_tokens=False
        )

        print(
            f"{rank + 1}: "
            f"id={token_id:3d} "
            f"token={repr(token_text):10s} "
            f"prob={probability:.6f}"
        )

    print()

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    generated_ids = model.generate(
        token_ids,
        max_new_tokens=20,
        eos_id=tokenizer.eos_id
    )

    print("Generated IDs")
    print("-------------")

    print(
        generated_ids
    )

    print()

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    print("Generated Text")
    print("--------------")

    print(
        repr(
            generated_text
        )
    )

    print()

    # --------------------------------------------------------
    # GPU Memory Information
    # --------------------------------------------------------

    print(
        "================================"
    )

    print(
        "Virtual GPU Memory"
    )

    print(
        "================================"
    )

    model.runtime.memory.info()

    print()

    # --------------------------------------------------------
    # Important Note
    # --------------------------------------------------------

    print(
        "NOTE:"
    )

    print(
        "This model has not been trained yet."
    )

    print(
        "The generated output is therefore random-like."
    )

    print(
        "Once end-to-end autograd is connected, "
        "the same demo can load trained parameters."
    )


if __name__ == "__main__":

    main()
