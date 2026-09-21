# layers.py
#
# Basic neural-network layers for the homemade LLM.
# v0.3 adds manual backward propagation for full-model training.

import math
import random
from typing import List, Optional

from tensor import Tensor, TensorRuntime, get_default_runtime
from autograd import Parameter


def _set_parameter_grad(parameter: Parameter, grad: Tensor) -> None:
    if grad.shape != parameter.shape:
        raise ValueError(
            f"Gradient shape mismatch: {grad.shape} != {parameter.shape}"
        )
    parameter.grad = grad


class Layer:
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad_output: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> List[Parameter]:
        return []


class Linear(Layer):
    """Fully connected layer: Y = X @ W + b."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        runtime: Optional[TensorRuntime] = None,
        seed: int = 0,
        init_scale: float = 0.02,
    ):
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Linear dimensions must be greater than 0.")

        self.in_features = in_features
        self.out_features = out_features
        self.runtime = runtime or get_default_runtime()
        self._last_input: Optional[Tensor] = None

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
            if bias else None
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.runtime is not self.runtime:
            raise ValueError("Tensor runtime does not match Linear runtime.")
        if len(x.shape) != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"Linear expects (*, {self.in_features}), got {x.shape}"
            )

        self._last_input = x
        output = x @ self.weight.tensor

        if self.bias is not None:
            rows, cols = output.shape
            bias_values = self.bias.tensor.flat()
            for row in range(rows):
                base = output.address + row * cols
                values = self.runtime.memory.read(base, cols)
                self.runtime.memory.write(
                    base,
                    [values[col] + bias_values[col] for col in range(cols)],
                )

        return output

    def backward(self, grad_output: Tensor) -> Tensor:
        if self._last_input is None:
            raise RuntimeError("Linear.backward() called before forward().")
        if grad_output.shape[1] != self.out_features:
            raise ValueError("Linear backward output-gradient mismatch.")

        x = self._last_input
        grad_input = grad_output @ self.weight.tensor.T
        grad_weight = x.T @ grad_output
        _set_parameter_grad(self.weight, grad_weight)

        if self.bias is not None:
            rows, cols = grad_output.shape
            values = [0.0] * cols
            flat = grad_output.flat()
            for row in range(rows):
                for col in range(cols):
                    values[col] += flat[row * cols + col]
            grad_bias = Tensor(values, runtime=self.runtime)
            _set_parameter_grad(self.bias, grad_bias)

        return grad_input

    def parameters(self) -> List[Parameter]:
        result = [self.weight]
        if self.bias is not None:
            result.append(self.bias)
        return result


class ReLU(Layer):
    def __init__(self):
        self._last_input: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        self._last_input = x
        result = Tensor.zeros(x.shape, runtime=x.runtime)
        x.runtime.memory.write(
            result.address,
            [value if value > 0.0 else 0.0 for value in x.flat()],
        )
        return result

    def backward(self, grad_output: Tensor) -> Tensor:
        if self._last_input is None:
            raise RuntimeError("ReLU.backward() called before forward().")
        x_values = self._last_input.flat()
        g_values = grad_output.flat()
        values = [
            g if x > 0.0 else 0.0
            for x, g in zip(x_values, g_values)
        ]
        return Tensor(
            _reshape(values, grad_output.shape),
            runtime=grad_output.runtime,
        )


class GELU(Layer):
    def __init__(self):
        self._last_input: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        self._last_input = x
        result = Tensor.zeros(x.shape, runtime=x.runtime)
        coefficient = math.sqrt(2.0 / math.pi)
        output = []
        for value in x.flat():
            inner = coefficient * (value + 0.044715 * value ** 3)
            output.append(0.5 * value * (1.0 + math.tanh(inner)))
        x.runtime.memory.write(result.address, output)
        return result

    def backward(self, grad_output: Tensor) -> Tensor:
        if self._last_input is None:
            raise RuntimeError("GELU.backward() called before forward().")

        coefficient = math.sqrt(2.0 / math.pi)
        result = []
        for x, upstream in zip(self._last_input.flat(), grad_output.flat()):
            inner = coefficient * (x + 0.044715 * x ** 3)
            t = math.tanh(inner)
            d_inner = coefficient * (1.0 + 3.0 * 0.044715 * x * x)
            derivative = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * d_inner
            result.append(upstream * derivative)

        return Tensor(
            _reshape(result, grad_output.shape),
            runtime=grad_output.runtime,
        )


class LayerNorm(Layer):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        runtime: Optional[TensorRuntime] = None,
    ):
        if normalized_shape <= 0:
            raise ValueError("normalized_shape must be > 0.")

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

        self._last_xhat = None
        self._last_inv_std = None
        self._last_shape = None

    def forward(self, x: Tensor) -> Tensor:
        if x.runtime is not self.runtime:
            raise ValueError("Tensor runtime does not match LayerNorm runtime.")
        if x.shape[-1] != self.normalized_shape:
            raise ValueError("LayerNorm dimension mismatch.")

        result = Tensor.zeros(x.shape, runtime=self.runtime)
        vector_size = self.normalized_shape
        vector_count = x.size // vector_size

        xhat_rows = []
        inv_stds = []
        gamma = self.gamma.tensor.flat()
        beta = self.beta.tensor.flat()

        for vector_index in range(vector_count):
            base = x.address + vector_index * vector_size
            values = self.runtime.memory.read(base, vector_size)
            mean = sum(values) / vector_size
            variance = sum((value - mean) ** 2 for value in values) / vector_size
            inv_std = 1.0 / math.sqrt(variance + self.eps)
            xhat = [(value - mean) * inv_std for value in values]

            out = [
                xhat[i] * gamma[i] + beta[i]
                for i in range(vector_size)
            ]
            self.runtime.memory.write(
                result.address + vector_index * vector_size,
                out,
            )
            xhat_rows.append(xhat)
            inv_stds.append(inv_std)

        self._last_xhat = xhat_rows
        self._last_inv_std = inv_stds
        self._last_shape = x.shape
        return result

    def backward(self, grad_output: Tensor) -> Tensor:
        if self._last_xhat is None or self._last_inv_std is None:
            raise RuntimeError("LayerNorm.backward() called before forward().")

        n = self.normalized_shape
        vector_count = grad_output.size // n
        gamma = self.gamma.tensor.flat()
        grad_flat = grad_output.flat()

        grad_input_rows = []
        grad_gamma = [0.0] * n
        grad_beta = [0.0] * n

        for row in range(vector_count):
            dy = grad_flat[row * n:(row + 1) * n]
            xhat = self._last_xhat[row]
            inv_std = self._last_inv_std[row]

            dxhat = [dy[i] * gamma[i] for i in range(n)]
            sum_dxhat = sum(dxhat)
            sum_dxhat_xhat = sum(dxhat[i] * xhat[i] for i in range(n))

            dx = [
                (inv_std / n)
                * (n * dxhat[i] - sum_dxhat - xhat[i] * sum_dxhat_xhat)
                for i in range(n)
            ]
            grad_input_rows.append(dx)

            for i in range(n):
                grad_gamma[i] += dy[i] * xhat[i]
                grad_beta[i] += dy[i]

        _set_parameter_grad(
            self.gamma,
            Tensor(grad_gamma, runtime=self.runtime),
        )
        _set_parameter_grad(
            self.beta,
            Tensor(grad_beta, runtime=self.runtime),
        )

        return Tensor(grad_input_rows, runtime=self.runtime)

    def parameters(self) -> List[Parameter]:
        return [self.gamma, self.beta]


class Softmax(Layer):
    def __init__(self):
        self._last_output: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) == 0:
            raise ValueError("Softmax requires a non-scalar Tensor.")

        last_dim = x.shape[-1]
        result = Tensor.zeros(x.shape, runtime=x.runtime)
        vector_count = x.size // last_dim

        for vector_index in range(vector_count):
            src = x.address + vector_index * last_dim
            dst = result.address + vector_index * last_dim
            values = x.runtime.memory.read(src, last_dim)
            max_value = max(values)
            exp_values = [math.exp(value - max_value) for value in values]
            total = sum(exp_values)
            x.runtime.memory.write(dst, [value / total for value in exp_values])

        self._last_output = result
        return result

    def backward(self, grad_output: Tensor) -> Tensor:
        if self._last_output is None:
            raise RuntimeError("Softmax.backward() called before forward().")

        last_dim = self._last_output.shape[-1]
        vector_count = self._last_output.size // last_dim
        y = self._last_output.flat()
        g = grad_output.flat()
        result = []

        for row in range(vector_count):
            start = row * last_dim
            yrow = y[start:start + last_dim]
            grow = g[start:start + last_dim]
            dot = sum(a * b for a, b in zip(grow, yrow))
            result.extend([
                yrow[i] * (grow[i] - dot)
                for i in range(last_dim)
            ])

        return Tensor(
            _reshape(result, self._last_output.shape),
            runtime=grad_output.runtime,
        )


class Sequential(Layer):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad_output: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def parameters(self) -> List[Parameter]:
        result = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                result.extend(layer.parameters())
        return result


def _reshape(values, shape):
    if len(shape) == 1:
        return list(values)
    if len(shape) == 2:
        rows, cols = shape
        return [
            list(values[row * cols:(row + 1) * cols])
            for row in range(rows)
        ]
    raise ValueError("v0.3 helper currently supports 1-D/2-D tensors.")


if __name__ == "__main__":
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    layer = Linear(2, 3, runtime=x.runtime, seed=42)
    y = layer(x)
    dy = Tensor.ones(y.shape, runtime=x.runtime)
    dx = layer.backward(dy)
    print("Y:", y)
    print("dX:", dx)
