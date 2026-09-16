# llm.py
#
# Minimal language model built from the homemade components.
#
# Dependencies:
#   tensor.py
#   embedding.py
#   transformer.py
#   layers.py
#
# Flow:
#
#   token IDs
#       |
#       v
#   Embedding
#       |
#       v
#   Transformer
#       |
#       v
#   Linear(d_model -> vocab_size)
#       |
#       v
#   Logits
#
# NumPy is intentionally not used.

from typing import Optional, List

from tensor import (
    Tensor,
    TensorRuntime,
    get_default_runtime,
)

from embedding import (
    Embedding,
)

from transformer import (
    Transformer,
)

from layers import (
    Linear,
    Softmax,
)


class LanguageModel:
    """
    Minimal Transformer-based language model.

    Parameters
    ----------
    vocab_size:
        Number of tokens in vocabulary.

    d_model:
        Embedding / Transformer dimension.

    num_layers:
        Number of Transformer blocks.

    hidden_dim:
        Feed-forward hidden size.
        Default: 4 * d_model

    causal:
        Whether to use causal self-attention.

    runtime:
        Optional TensorRuntime.

    seed:
        Base random seed.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        hidden_dim: Optional[int] = None,
        causal: bool = True,
        runtime: Optional[TensorRuntime] = None,
        seed: int = 0,
    ):

        if vocab_size <= 0:
            raise ValueError(
                "vocab_size must be greater than 0."
            )

        if d_model <= 0:
            raise ValueError(
                "d_model must be greater than 0."
            )

        if num_layers <= 0:
            raise ValueError(
                "num_layers must be greater than 0."
            )

        if hidden_dim is None:
            hidden_dim = 4 * d_model

        if runtime is None:
            runtime = get_default_runtime()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.causal = causal
        self.runtime = runtime

        self.embedding = Embedding(
            vocab_size=vocab_size,
            embedding_dim=d_model,
            runtime=self.runtime,
            seed=seed + 1
        )

        self.transformer = Transformer(
            num_layers=num_layers,
            d_model=d_model,
            hidden_dim=hidden_dim,
            causal=causal,
            runtime=self.runtime,
            seed=seed + 100
        )

        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            bias=False,
            runtime=self.runtime,
            seed=seed + 1000
        )

        self.softmax = Softmax()

    def forward(
        self,
        token_ids: List[int]
    ) -> Tensor:
        """
        Compute vocabulary logits.

        Input:
            token_ids:
                list of token IDs

        Output:
            Tensor shape:
                [sequence_length, vocab_size]
        """

        if not isinstance(token_ids, list):
            raise TypeError(
                "token_ids must be a list."
            )

        if len(token_ids) == 0:
            raise ValueError(
                "token_ids must not be empty."
            )

        x = self.embedding(token_ids)
        x = self.transformer(x)
        logits = self.lm_head(x)
        return logits

    def __call__(
        self,
        token_ids: List[int]
    ) -> Tensor:
        return self.forward(token_ids)

    def probabilities(
        self,
        token_ids: List[int]
    ) -> Tensor:
        logits = self.forward(token_ids)
        return self.softmax(logits)

    def last_logits(
        self,
        token_ids: List[int]
    ) -> List[float]:
        logits = self.forward(token_ids)
        sequence_length = logits.shape[0]
        vocab_size = logits.shape[1]

        start_address = (
            logits.address
            + (sequence_length - 1) * vocab_size
        )

        return self.runtime.memory.read(
            start_address,
            vocab_size
        )

    def last_probabilities(
        self,
        token_ids: List[int]
    ) -> List[float]:
        logits = self.last_logits(token_ids)

        import math

        max_value = max(logits)

        exp_values = [
            math.exp(value - max_value)
            for value in logits
        ]

        total = sum(exp_values)

        return [
            value / total
            for value in exp_values
        ]

    def next_token(
        self,
        token_ids: List[int]
    ) -> int:
        logits = self.last_logits(token_ids)

        best_index = 0
        best_value = logits[0]

        for index in range(1, len(logits)):
            if logits[index] > best_value:
                best_value = logits[index]
                best_index = index

        return best_index

    def generate(
        self,
        token_ids: List[int],
        max_new_tokens: int = 10,
        eos_id: Optional[int] = None,
    ) -> List[int]:
        """
        Greedy text generation.
        """

        if max_new_tokens < 0:
            raise ValueError(
                "max_new_tokens must be >= 0."
            )

        generated = list(token_ids)

        for _ in range(max_new_tokens):
            next_id = self.next_token(generated)
            generated.append(next_id)

            if (
                eos_id is not None
                and next_id == eos_id
            ):
                break

        return generated

    def info(self) -> None:
        print("Language Model")
        print("==============")
        print(f"Vocabulary size : {self.vocab_size}")
        print(f"d_model         : {self.d_model}")
        print(f"Layers          : {self.num_layers}")
        print(f"FFN hidden dim  : {self.hidden_dim}")
        print(f"Causal          : {self.causal}")


if __name__ == "__main__":

    print()
    print("================================")
    print("Language Model Test")
    print("================================")
    print()

    vocab_size = 16
    d_model = 8
    num_layers = 2

    model = LanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        hidden_dim=32,
        seed=42
    )

    model.info()
    print()

    token_ids = [1, 4, 7, 2]

    print("Input token IDs:")
    print(token_ids)
    print()

    logits = model(token_ids)

    print("Logits:")
    print(logits)
    print()

    print("Logits shape:")
    print(logits.shape)
    print()

    assert logits.shape == (
        len(token_ids),
        vocab_size
    )

    print("Logits shape check: OK")
    print()

    next_id = model.next_token(token_ids)

    print("Predicted next token ID:")
    print(next_id)
    print()

    probabilities = model.last_probabilities(
        token_ids
    )

    print("Next-token probabilities:")
    print(probabilities)
    print()

    print("Probability sum:")
    print(sum(probabilities))
    print()

    generated = model.generate(
        token_ids,
        max_new_tokens=5
    )

    print("Generated token IDs:")
    print(generated)
    print()

    print("================================")
    print("Virtual GPU Memory")
    print("================================")

    model.runtime.memory.info()
