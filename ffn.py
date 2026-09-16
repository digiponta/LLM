# ffn.py
#
# Feed Forward Network for the homemade Transformer.
#
# Purpose:
#   - Position-wise feed-forward network
#   - Used inside Transformer blocks
#
# Dependencies:
#   tensor.py
#   layers.py
#
# Structure:
#
#   Input
#     |
#     v
#   Linear(d_model -> hidden_dim)
#     |
#     v
#   GELU
#     |
#     v
#   Linear(hidden_dim -> d_model)
#     |
#     v
#   Output
#
# NumPy is intentionally not used.

from typing import Optional

from tensor import (
    Tensor,
    TensorRuntime,
    get_default_runtime,
)

from layers import (
    Linear,
    GELU,
)


class FeedForward:
    """
    Transformer feed-forward network.

    Parameters
    ----------
    d_model:
        Embedding dimension.

    hidden_dim:
        Hidden dimension of the FFN.
        A common choice is 4 * d_model.

    runtime:
        Optional TensorRuntime.

    seed:
        Base random seed.

    bias:
        Whether Linear layers use bias.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        runtime: Optional[TensorRuntime] = None,
        seed: int = 0,
        bias: bool = True,
    ):
        if d_model <= 0:
            raise ValueError("d_model must be greater than 0.")

        if hidden_dim is None:
            hidden_dim = 4 * d_model

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than 0.")

        self.d_model = d_model
        self.hidden_dim = hidden_dim

        if runtime is None:
            runtime = get_default_runtime()

        self.runtime = runtime

        self.fc1 = Linear(
            in_features=d_model,
            out_features=hidden_dim,
            bias=bias,
            runtime=self.runtime,
            seed=seed + 1,
        )

        self.activation = GELU()

        self.fc2 = Linear(
            in_features=hidden_dim,
            out_features=d_model,
            bias=bias,
            runtime=self.runtime,
            seed=seed + 2,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Input shape:
            [sequence_length, d_model]

        Output shape:
            [sequence_length, d_model]
        """
        if x.runtime is not self.runtime:
            raise ValueError(
                "Tensor runtime does not match FeedForward runtime."
            )

        if len(x.shape) != 2:
            raise ValueError(
                "FeedForward currently expects a 2-D Tensor."
            )

        if x.shape[1] != self.d_model:
            raise ValueError(
                "FeedForward input dimension mismatch: "
                f"expected {self.d_model}, got {x.shape[1]}"
            )

        hidden = self.fc1(x)
        hidden = self.activation(hidden)
        output = self.fc2(hidden)
        return output

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def info(self) -> None:
        print("Feed Forward Network")
        print("====================")
        print(f"d_model    : {self.d_model}")
        print(f"hidden_dim : {self.hidden_dim}")
        print(f"fc1 weight : {self.fc1.weight.shape}")
        print(f"fc2 weight : {self.fc2.weight.shape}")


if __name__ == "__main__":
    print()
    print("================================")
    print("Feed Forward Network Test")
    print("================================")
    print()

    x = Tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])

    ffn = FeedForward(
        d_model=4,
        hidden_dim=16,
        runtime=x.runtime,
        seed=42
    )

    ffn.info()
    print()
    print("Input:")
    print(x)
    print()
    print("Input shape:")
    print(x.shape)
    print()

    output = ffn(x)

    print("FFN output:")
    print(output)
    print()
    print("Output shape:")
    print(output.shape)
    print()

    assert output.shape == x.shape
    print("Shape check: OK")
    print()

    print("================================")
    print("Virtual GPU Memory")
    print("================================")
    x.runtime.memory.info()
