# transformer.py
#
# Transformer Block for the homemade LLM.
#
# Dependencies:
#   tensor.py
#   layers.py
#   attention.py
#   ffn.py
#
# Structure:
#
#   x
#   |
#   +------------------------+
#   |                        |
#   v                        |
# LayerNorm                  |
#   |                        |
#   v                        |
# SelfAttention              |
#   |                        |
#   +---------- Add <--------+
#              |
#              v
#          LayerNorm
#              |
#              v
#             FFN
#              |
#   +---------- Add <--------+
#   |
#   v
# Output
#
# Pre-Norm Transformer is used.
#
# NumPy is intentionally not used.

from typing import Optional, List

from tensor import (
    Tensor,
    TensorRuntime,
    get_default_runtime,
)

from layers import (
    LayerNorm,
)

from attention import (
    SelfAttention,
)

from ffn import (
    FeedForward,
)


class TransformerBlock:
    """
    Single Transformer block using a Pre-Norm architecture.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        causal: bool = True,
        runtime: Optional[TensorRuntime] = None,
        seed: int = 0,
    ):

        if d_model <= 0:
            raise ValueError(
                "d_model must be greater than 0."
            )

        if hidden_dim is None:
            hidden_dim = 4 * d_model

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be greater than 0."
            )

        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.causal = causal

        if runtime is None:
            runtime = get_default_runtime()

        self.runtime = runtime

        self.norm1 = LayerNorm(
            normalized_shape=d_model,
            runtime=self.runtime
        )

        self.attention = SelfAttention(
            d_model=d_model,
            causal=causal,
            runtime=self.runtime,
            seed=seed + 10
        )

        self.norm2 = LayerNorm(
            normalized_shape=d_model,
            runtime=self.runtime
        )

        self.ffn = FeedForward(
            d_model=d_model,
            hidden_dim=hidden_dim,
            runtime=self.runtime,
            seed=seed + 20
        )

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
                "Tensor runtime does not match TransformerBlock runtime."
            )

        if len(x.shape) != 2:
            raise ValueError(
                "TransformerBlock currently expects a 2-D Tensor."
            )

        if x.shape[1] != self.d_model:
            raise ValueError(
                "TransformerBlock dimension mismatch: "
                f"expected {self.d_model}, got {x.shape[1]}"
            )

        normalized = self.norm1(x)
        attention_output = self.attention(normalized)
        x = x + attention_output

        normalized = self.norm2(x)
        ffn_output = self.ffn(normalized)
        x = x + ffn_output

        return x

    def __call__(
        self,
        x: Tensor
    ) -> Tensor:
        return self.forward(x)

    def info(self) -> None:
        print("Transformer Block")
        print("=================")
        print(f"d_model    : {self.d_model}")
        print(f"hidden_dim : {self.hidden_dim}")
        print(f"causal     : {self.causal}")


class Transformer:
    """
    Stack of Transformer blocks.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        hidden_dim: Optional[int] = None,
        causal: bool = True,
        runtime: Optional[TensorRuntime] = None,
        seed: int = 0,
        final_norm: bool = True,
    ):

        if num_layers <= 0:
            raise ValueError(
                "num_layers must be greater than 0."
            )

        if d_model <= 0:
            raise ValueError(
                "d_model must be greater than 0."
            )

        if hidden_dim is None:
            hidden_dim = 4 * d_model

        if runtime is None:
            runtime = get_default_runtime()

        self.num_layers = num_layers
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.causal = causal
        self.runtime = runtime

        self.blocks: List[TransformerBlock] = []

        for layer_index in range(num_layers):
            block = TransformerBlock(
                d_model=d_model,
                hidden_dim=hidden_dim,
                causal=causal,
                runtime=self.runtime,
                seed=seed + layer_index * 100
            )
            self.blocks.append(block)

        if final_norm:
            self.final_norm = LayerNorm(
                normalized_shape=d_model,
                runtime=self.runtime
            )
        else:
            self.final_norm = None

    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        """
        Run input through all Transformer blocks.
        """

        if x.runtime is not self.runtime:
            raise ValueError(
                "Tensor runtime does not match Transformer runtime."
            )

        if len(x.shape) != 2:
            raise ValueError(
                "Transformer expects a 2-D Tensor."
            )

        if x.shape[1] != self.d_model:
            raise ValueError(
                "Transformer input dimension mismatch."
            )

        for block in self.blocks:
            x = block(x)

        if self.final_norm is not None:
            x = self.final_norm(x)

        return x

    def __call__(
        self,
        x: Tensor
    ) -> Tensor:
        return self.forward(x)

    def info(self) -> None:
        print("Transformer")
        print("===========")
        print(f"Layers     : {self.num_layers}")
        print(f"d_model    : {self.d_model}")
        print(f"hidden_dim : {self.hidden_dim}")
        print(f"causal     : {self.causal}")
        print(f"final norm : {self.final_norm is not None}")


if __name__ == "__main__":

    print()
    print("================================")
    print("Transformer Test")
    print("================================")
    print()

    x = Tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ])

    print("Input:")
    print(x)
    print()
    print("Input shape:")
    print(x.shape)
    print()

    print("================================")
    print("Single Transformer Block")
    print("================================")

    block = TransformerBlock(
        d_model=8,
        hidden_dim=32,
        causal=True,
        runtime=x.runtime,
        seed=42
    )

    block.info()
    print()

    block_output = block(x)

    print("Block output:")
    print(block_output)
    print()
    print("Block output shape:")
    print(block_output.shape)
    print()

    assert block_output.shape == x.shape
    print("Block shape check: OK")
    print()

    print("================================")
    print("Transformer Stack")
    print("================================")

    model = Transformer(
        num_layers=2,
        d_model=8,
        hidden_dim=32,
        causal=True,
        runtime=x.runtime,
        seed=100
    )

    model.info()
    print()

    output = model(x)

    print("Transformer output:")
    print(output)
    print()
    print("Output shape:")
    print(output.shape)
    print()

    assert output.shape == x.shape
    print("Transformer shape check: OK")
    print()

    print("================================")
    print("Virtual GPU Memory")
    print("================================")

    x.runtime.memory.info()
