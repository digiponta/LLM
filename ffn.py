# ffn.py
#
# Feed Forward Network with v0.3 backward propagation.

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
            d_model, hidden_dim, bias=bias,
            runtime=self.runtime, seed=seed + 1,
        )
        self.activation = GELU()
        self.fc2 = Linear(
            hidden_dim, d_model, bias=bias,
            runtime=self.runtime, seed=seed + 2,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.runtime is not self.runtime:
            raise ValueError("Tensor runtime does not match FeedForward runtime.")
        if len(x.shape) != 2 or x.shape[1] != self.d_model:
            raise ValueError("FeedForward input dimension mismatch.")
        return self.fc2(self.activation(self.fc1(x)))

    def backward(self, grad_output: Tensor) -> Tensor:
        grad = self.fc2.backward(grad_output)
        grad = self.activation.backward(grad)
        return self.fc1.backward(grad)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> List[Parameter]:
        return self.fc1.parameters() + self.fc2.parameters()


if __name__ == "__main__":
    x = Tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
    ffn = FeedForward(4, hidden_dim=16, runtime=x.runtime, seed=42)
    y = ffn(x)
    dy = Tensor.ones(y.shape, runtime=x.runtime)
    dx = ffn.backward(dy)
    print("Y:", y)
    print("dX:", dx)
