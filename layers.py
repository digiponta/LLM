# layers.py
#
# Basic neural-network layers for the homemade LLM.
#
# Purpose:
#   - Linear
#   - LayerNorm
#   - ReLU
#   - GELU
#   - Softmax
#
# Dependencies:
#   tensor.py
#
# NumPy is intentionally not used.

import math
import random
from typing import Optional

from tensor import (
    Tensor,
    TensorRuntime,
    get_default_runtime,
)


class Layer:
    """Base class for neural-network layers."""

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


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

        if runtime is None:
            runtime = get_default_runtime()
        self.runtime = runtime

        rng = random.Random(seed)
        weights = []
        for _ in range(in_features):
            row = []
            for _ in range(out_features):
                row.append(rng.uniform(-init_scale, init_scale))
            weights.append(row)

        self.weight = Tensor(weights, runtime=self.runtime)

        if bias:
            self.bias = Tensor.zeros((out_features,), runtime=self.runtime)
        else:
            self.bias = None

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

        output = x @ self.weight

        if self.bias is not None:
            rows = output.shape[0]
            cols = output.shape[1]
            for row in range(rows):
                for col in range(cols):
                    output_address = output.address + row * cols + col
                    bias_address = self.bias.address + col
                    value = self.runtime.memory.read_scalar(output_address)
                    bias_value = self.runtime.memory.read_scalar(bias_address)
                    self.runtime.memory.write_scalar(
                        output_address,
                        value + bias_value,
                    )

        return output

    def info(self) -> None:
        print("Linear Layer")
        print("============")
        print(f"Input features  : {self.in_features}")
        print(f"Output features : {self.out_features}")
        print(f"Weight shape    : {self.weight.shape}")
        print(f"Bias            : {self.bias is not None}")


class ReLU(Layer):
    """Rectified Linear Unit."""

    def forward(self, x: Tensor) -> Tensor:
        result = Tensor.zeros(x.shape, runtime=x.runtime)
        values = x.flat()
        output = [value if value > 0.0 else 0.0 for value in values]
        x.runtime.memory.write(result.address, output)
        return result


class GELU(Layer):
    """Gaussian Error Linear Unit using the tanh approximation."""

    def forward(self, x: Tensor) -> Tensor:
        result = Tensor.zeros(x.shape, runtime=x.runtime)
        values = x.flat()
        output = []
        coefficient = math.sqrt(2.0 / math.pi)

        for value in values:
            inner = coefficient * (
                value + 0.044715 * value * value * value
            )
            gelu_value = 0.5 * value * (1.0 + math.tanh(inner))
            output.append(gelu_value)

        x.runtime.memory.write(result.address, output)
        return result


class LayerNorm(Layer):
    """Layer Normalization over the final dimension."""

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

        if runtime is None:
            runtime = get_default_runtime()
        self.runtime = runtime

        self.gamma = Tensor.ones((normalized_shape,), runtime=self.runtime)
        self.beta = Tensor.zeros((normalized_shape,), runtime=self.runtime)

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
            variance = 0.0
            for value in values:
                diff = value - mean
                variance += diff * diff
            variance /= vector_size

            inv_std = 1.0 / math.sqrt(variance + self.eps)

            for i in range(vector_size):
                normalized = (values[i] - mean) * inv_std
                gamma = self.runtime.memory.read_scalar(self.gamma.address + i)
                beta = self.runtime.memory.read_scalar(self.beta.address + i)
                value = normalized * gamma + beta
                self.runtime.memory.write_scalar(out_base + i, value)

        return result


class Softmax(Layer):
    """Softmax over the final Tensor dimension."""

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) == 0:
            raise ValueError("Softmax requires a non-scalar Tensor.")

        last_dim = x.shape[-1]
        if last_dim <= 0:
            raise ValueError("Softmax dimension must be > 0.")

        vector_count = x.size // last_dim
        result = Tensor.zeros(x.shape, runtime=x.runtime)

        for vector_index in range(vector_count):
            src_address = x.address + vector_index * last_dim
            dst_address = result.address + vector_index * last_dim
            values = x.runtime.memory.read(src_address, last_dim)

            max_value = max(values)
            exp_values = [math.exp(value - max_value) for value in values]
            total = sum(exp_values)

            if total == 0.0:
                raise ZeroDivisionError("Softmax normalization sum is zero.")

            output = [value / total for value in exp_values]
            x.runtime.memory.write(dst_address, output)

        return result


class Sequential(Layer):
    """Apply several layers in sequence."""

    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    print()
    print("================================")
    print("Neural Network Layers Test")
    print("================================")
    print()

    x = Tensor([
        [1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0],
    ])

    print("Input:")
    print(x)
    print()

    print("================================")
    print("Linear")
    print("================================")

    linear = Linear(
        in_features=4,
        out_features=3,
        runtime=x.runtime,
        seed=42,
    )

    y = linear(x)
    print("Linear output:")
    print(y)
    print("Shape:", y.shape)
    print()

    print("================================")
    print("ReLU")
    print("================================")

    relu_input = Tensor([
        [-2.0, -1.0, 0.0],
        [1.0, 2.0, 3.0],
    ])

    relu = ReLU()
    relu_output = relu(relu_input)
    print(relu_output)
    print()

    print("================================")
    print("GELU")
    print("================================")

    gelu = GELU()
    gelu_output = gelu(relu_input)
    print(gelu_output)
    print()

    print("================================")
    print("LayerNorm")
    print("================================")

    norm = LayerNorm(
        normalized_shape=4,
        runtime=x.runtime,
    )

    norm_output = norm(x)
    print(norm_output)
    print()

    print("================================")
    print("Softmax")
    print("================================")

    softmax_input = Tensor([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 1.0],
    ])

    softmax = Softmax()
    softmax_output = softmax(softmax_input)
    print(softmax_output)
    print()

    values = softmax_output.tolist()
    print("Row 0 sum:", sum(values[0]))
    print("Row 1 sum:", sum(values[1]))
    print()

    print("================================")
    print("Simple FFN")
    print("================================")

    ffn = Sequential(
        Linear(4, 8, runtime=x.runtime, seed=10),
        GELU(),
        Linear(8, 4, runtime=x.runtime, seed=20),
    )

    ffn_output = ffn(x)
    print(ffn_output)
    print("FFN output shape:", ffn_output.shape)
    print()

    print("================================")
    print("Virtual GPU Memory")
    print("================================")

    x.runtime.memory.info()
