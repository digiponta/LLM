# attention.py
#
# Simple Self-Attention layer
#
# Purpose:
#   - Compute Query, Key, Value
#   - Scaled dot-product attention
#   - Optional causal mask
#
# Dependencies:
#   tensor.py
#   layers.py
#
# Flow:
#
#   X
#   |
#   +--> Linear --> Q
#   |
#   +--> Linear --> K
#   |
#   +--> Linear --> V
#
#   scores = Q @ K.T
#   scores = scores / sqrt(d_model)
#   scores = mask(scores)
#   probs  = softmax(scores)
#   out    = probs @ V
#
# NumPy is intentionally not used.

import math
from typing import Optional

from tensor import (
    Tensor,
    TensorRuntime,
    get_default_runtime,
)

from layers import (
    Linear,
    Softmax,
)


class SelfAttention:
    """
    Single-head self-attention.

    Parameters
    ----------
    d_model:
        Embedding dimension.

    causal:
        If True, apply causal mask so token i
        cannot attend to future tokens.

    runtime:
        Tensor runtime.

    seed:
        Base random seed for Q/K/V projection weights.
    """

    def __init__(
        self,
        d_model: int,
        causal: bool = True,
        runtime: Optional[TensorRuntime] = None,
        seed: int = 0,
    ):

        if d_model <= 0:
            raise ValueError(
                "d_model must be greater than 0."
            )

        self.d_model = d_model
        self.causal = causal

        if runtime is None:
            runtime = get_default_runtime()

        self.runtime = runtime

        self.q_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False,
            runtime=self.runtime,
            seed=seed + 1
        )

        self.k_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False,
            runtime=self.runtime,
            seed=seed + 2
        )

        self.v_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False,
            runtime=self.runtime,
            seed=seed + 3
        )

        self.out_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False,
            runtime=self.runtime,
            seed=seed + 4
        )

        self.softmax = Softmax()

    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        """
        Input shape:
            [sequence_length, d_model]

        Output shape:
            [sequence_length, d_model]
        """

        if x.runtime is not self.runtime:
            raise ValueError(
                "Tensor runtime does not match "
                "SelfAttention runtime."
            )

        if len(x.shape) != 2:
            raise ValueError(
                "SelfAttention currently expects "
                "a 2-D Tensor."
            )

        embedding_dim = x.shape[1]

        if embedding_dim != self.d_model:
            raise ValueError(
                "SelfAttention dimension mismatch: "
                f"expected {self.d_model}, "
                f"got {embedding_dim}"
            )

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = q @ k.T

        scale = math.sqrt(
            float(self.d_model)
        )

        scores = scores / scale

        if self.causal:
            scores = self._apply_causal_mask(
                scores
            )

        attention_weights = self.softmax(
            scores
        )

        context = (
            attention_weights
            @ v
        )

        output = self.out_proj(
            context
        )

        return output

    def __call__(
        self,
        x: Tensor
    ) -> Tensor:

        return self.forward(
            x
        )

    def _apply_causal_mask(
        self,
        scores: Tensor
    ) -> Tensor:
        """
        Apply triangular causal mask.
        """

        if len(scores.shape) != 2:
            raise ValueError(
                "Causal mask requires 2-D scores."
            )

        rows, cols = scores.shape

        if rows != cols:
            raise ValueError(
                "Causal attention score matrix "
                "must be square."
            )

        result = scores.clone()

        negative_large = -1.0e9

        for row in range(rows):
            for col in range(cols):
                if col > row:
                    address = (
                        result.address
                        + row * cols
                        + col
                    )

                    self.runtime.memory.write_scalar(
                        address,
                        negative_large
                    )

        return result

    def get_attention_weights(
        self,
        x: Tensor
    ) -> Tensor:
        """
        Return only attention probabilities.
        """

        if x.runtime is not self.runtime:
            raise ValueError(
                "Tensor runtime mismatch."
            )

        q = self.q_proj(x)
        k = self.k_proj(x)

        scores = (
            q
            @ k.T
        )

        scores = (
            scores
            / math.sqrt(
                float(self.d_model)
            )
        )

        if self.causal:
            scores = self._apply_causal_mask(
                scores
            )

        return self.softmax(
            scores
        )

    def info(self) -> None:

        print(
            "Self Attention"
        )

        print(
            "=============="
        )

        print(
            f"d_model : "
            f"{self.d_model}"
        )

        print(
            f"Causal  : "
            f"{self.causal}"
        )


if __name__ == "__main__":

    print()
    print(
        "================================"
    )
    print(
        "Self Attention Test"
    )
    print(
        "================================"
    )
    print()

    x = Tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])

    attention = SelfAttention(
        d_model=4,
        causal=True,
        runtime=x.runtime,
        seed=42
    )

    attention.info()

    print()
    print("Input:")
    print(x)
    print()

    weights = attention.get_attention_weights(x)

    print("Attention weights:")
    print(weights)
    print()
    print("Attention weight shape:")
    print(weights.shape)
    print()

    weight_rows = weights.tolist()

    for i, row in enumerate(weight_rows):
        print(
            f"Row {i} sum:",
            sum(row)
        )

    print()

    output = attention(x)

    print("Attention output:")
    print(output)
    print()
    print("Output shape:")
    print(output.shape)
    print()

    print("================================")
    print("Virtual GPU Memory")
    print("================================")

    x.runtime.memory.info()
