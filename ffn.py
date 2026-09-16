# ffn.py
#
# Feed Forward Network for the homemade Transformer.

from typing import List, Optional

from tensor import Tensor, TensorRuntime, get_default_runtime
from autograd import Parameter
from layers import Linear, GELU


class FeedForward:
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
        self.runtime = runtime or get_default_runtime()

        self.fc1 = Linear(
            d_model,
            hidden_dim,
            bias=bias,
            runtime=self.runtime,
            seed=seed + 1,
        )
        self.activation = GELU()
        self.fc2 = Linear(
            hidden_dim,
            d_model,
            bias=bias,
            runtime=self.runtime,
            seed=seed + 2,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.runtime is not self.runtime:
            raise ValueError("Tensor runtime does not match FeedForward runtime.")
        if len(x.shape) != 2:
            raise ValueError("FeedForward currently expects a 2-D Tensor.")
        if x.shape[1] != self.d_model:
            raise ValueError(
                f"FeedForward input dimension mismatch: expected {self.d_model}, "
                f"got {x.shape[1]}"
            )

        return self.fc2(self.activation(self.fc1(x)))

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> List[Parameter]:
        return self.fc1.parameters() + self.fc2.parameters()

    def info(self) -> None:
        print("Feed Forward Network")
        print("====================")
        print(f"d_model    : {self.d_model}")
        print(f"hidden_dim : {self.hidden_dim}")
        print(f"Parameters : {len(self.parameters())}")


if __name__ == "__main__":
    x = Tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    ffn = FeedForward(4, hidden_dim=16, runtime=x.runtime, seed=42)
    print(ffn(x))
    ffn.info()
