# layers.py
#
# Basic neural-network layers for the homemade LLM.
#
# Trainable state is stored as autograd.Parameter objects.
# Forward computation still uses the underlying Tensor objects.

import math
import random
from typing import List, Optional

from tensor import Tensor, TensorRuntime, get_default_runtime
from autograd import Parameter


class Layer:
    """Base class for neural-network layers."""

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> List[Parameter]:
        return []


class Linear(Layer):
    """Fully connected linear layer: Y = X @ W + b."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        runtime: Optional[TensorRuntime] = None,
        seed: int = 0,
        init_scale: float = 0.02,
    ):
        if in_features <= 0:
            raise ValueError("in_features must be greater than 0.")
        if out_features <= 0:
            raise ValueError("out_features must be greater than 0.")

        self.in_features = in_features
        self.out_features = out_features
        self.runtime = runtime or get_default_runtime()

        rng = random.Random(seed)
        weights = [
            [rng.uniform(-init_scale, init_scale) for _ in range(out_features)]
            for _ in range(in_features)
        ]

        self.weight = Parameter(
            Tensor(weights, runtime=self.runtime),
            name="weight",
        )

        self.bias = (
            Parameter(
                Tensor.zeros((out_features,), runtime=self.runtime),
                name="bias",
            )
            if bias
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.runtime is not self.runtime:
            raise ValueError("Tensor runtime does not match Linear runtime.")
        if len(x.shape) != 2:
            raise ValueError("Linear currently expects a 2-D Tensor.")
        if x.shape[1] != self.in_features:
            raise ValueError(
                "Linear input dimension mismatch: "
                f"expected {self.in_features}, got {x.shape[1]}"
            )

        output = x @ self.weight.tensor

        if self.bias is not None:
            rows, cols = output.shape
            for row in range(rows):
                for col in range(cols):
                    output_address = output.address + row * cols + col
                    value = self.runtime.memory.read_scalar(output_address)
                    bias_value = self.runtime.memory.read_scalar(
                        self.bias.address + col
                    )
                    self.runtime.memory.write_scalar(
                        output_address,
                        value + bias_value,
                    )

        return output

    def parameters(self) -> List[Parameter]:
        result = [self.weight]
        if self.bias is not None:
            result.append(self.bias)
        return result

    def info(self) -> None:
        print("Linear Layer")
        print("============")
        print(f"Input features  : {self.in_features}")
        print(f"Output features : {self.out_features}")
        print(f"Weight shape    : {self.weight.shape}")
        print(f"Bias            : {self.bias is not None}")


class ReLU(Layer):
    def forward(self, x: Tensor) -> Tensor:
        result = Tensor.zeros(x.shape, runtime=x.runtime)
        x.runtime.memory.write(
            result.address,
            [value if value > 0.0 else 0.0 for value in x.flat()],
        )
        return result


class GELU(Layer):
    def forward(self, x: Tensor) -> Tensor:
        result = Tensor.zeros(x.shape, runtime=x.runtime)
        coefficient = math.sqrt(2.0 / math.pi)
        output = []
        for value in x.flat():
            inner = coefficient * (value + 0.044715 * value ** 3)
            output.append(0.5 * value * (1.0 + math.tanh(inner)))
        x.runtime.memory.write(result.address, output)
        return result


class LayerNorm(Layer):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        runtime: Optional[TensorRuntime] = None,
    ):
        if normalized_shape <= 0:
            raise ValueError("normalized_shape must be > 0.")
        if eps <= 0:
            raise ValueError("eps must be > 0.")

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.runtime = runtime or get_default_runtime()

        self.gamma = Parameter(
            Tensor.ones((normalized_shape,), runtime=self.runtime),
            name="gamma",
        )
        self.beta = Parameter(
            Tensor.zeros((normalized_shape,), runtime=self.runtime),
            name="beta",
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.runtime is not self.runtime:
            raise ValueError("Tensor runtime does not match LayerNorm runtime.")
        if len(x.shape) == 0:
            raise ValueError("LayerNorm requires at least one dimension.")
        if x.shape[-1] != self.normalized_shape:
            raise ValueError(
                "LayerNorm dimension mismatch: "
                f"expected {self.normalized_shape}, got {x.shape[-1]}"
            )

        result = Tensor.zeros(x.shape, runtime=self.runtime)
        vector_size = self.normalized_shape
        vector_count = x.size // vector_size

        for vector_index in range(vector_count):
            base = x.address + vector_index * vector_size
            out_base = result.address + vector_index * vector_size
            values = self.runtime.memory.read(base, vector_size)
            mean = sum(values) / vector_size
            variance = sum((value - mean) ** 2 for value in values) / vector_size
            inv_std = 1.0 / math.sqrt(variance + self.eps)

            for i in range(vector_size):
                normalized = (values[i] - mean) * inv_std
                gamma = self.runtime.memory.read_scalar(self.gamma.address + i)
                beta = self.runtime.memory.read_scalar(self.beta.address + i)
                self.runtime.memory.write_scalar(
                    out_base + i,
                    normalized * gamma + beta,
                )

        return result

    def parameters(self) -> List[Parameter]:
        return [self.gamma, self.beta]


class Softmax(Layer):
    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) == 0:
            raise ValueError("Softmax requires a non-scalar Tensor.")

        last_dim = x.shape[-1]
        if last_dim <= 0:
            raise ValueError("Softmax dimension must be > 0.")

        result = Tensor.zeros(x.shape, runtime=x.runtime)
        vector_count = x.size // last_dim

        for vector_index in range(vector_count):
            src = x.address + vector_index * last_dim
            dst = result.address + vector_index * last_dim
            values = x.runtime.memory.read(src, last_dim)
            max_value = max(values)
            exp_values = [math.exp(value - max_value) for value in values]
            total = sum(exp_values)
            if total == 0.0:
                raise ZeroDivisionError("Softmax normalization sum is zero.")
            x.runtime.memory.write(dst, [value / total for value in exp_values])

        return result


class Sequential(Layer):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Parameter]:
        result = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                result.extend(layer.parameters())
        return result


if __name__ == "__main__":
    x = Tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    linear = Linear(4, 3, runtime=x.runtime, seed=42)
    y = linear(x)
    print("Linear output:", y)
    print("Parameter count:", len(linear.parameters()))
