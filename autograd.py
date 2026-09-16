# autograd.py
#
# Automatic differentiation engine for the homemade LLM.
#
# Dependencies:
#   tensor.py
#
# Purpose:
#   - Build a computation graph
#   - Store gradients
#   - Perform reverse-mode automatic differentiation
#
# Supported operations:
#   - add
#   - subtract
#   - multiply
#   - divide
#   - matrix multiplication
#   - transpose
#   - sum
#   - mean
#   - exp
#   - log
#   - tanh
#
# Notes:
#   - NumPy is intentionally not used.
#   - Broadcasting is not implemented yet.
#   - This is the first educational autograd implementation.

import math
from typing import (
    Callable,
    List,
    Optional,
    Set,
)

from tensor import (
    Tensor,
    TensorRuntime,
    get_default_runtime,
)


def zeros_like(tensor: Tensor) -> Tensor:
    return Tensor.zeros(tensor.shape, runtime=tensor.runtime)


def ones_like(tensor: Tensor) -> Tensor:
    return Tensor.ones(tensor.shape, runtime=tensor.runtime)


def fill_like(tensor: Tensor, value: float) -> Tensor:
    return Tensor.full(tensor.shape, value, runtime=tensor.runtime)


def add_inplace(destination: Tensor, source: Tensor) -> None:
    """destination += source"""
    if destination.shape != source.shape:
        raise ValueError("Gradient shape mismatch.")

    for i in range(destination.size):
        a = destination.runtime.memory.read_scalar(destination.address + i)
        b = source.runtime.memory.read_scalar(source.address + i)
        destination.runtime.memory.write_scalar(destination.address + i, a + b)


class AutoTensor:
    """Tensor with automatic differentiation metadata."""

    def __init__(
        self,
        tensor: Tensor,
        requires_grad: bool = False,
        name: Optional[str] = None,
    ):
        if not isinstance(tensor, Tensor):
            raise TypeError("AutoTensor requires Tensor.")

        self.tensor = tensor
        self.requires_grad = requires_grad
        self.name = name
        self.grad: Optional[Tensor] = None
        self.parents: List["AutoTensor"] = []
        self._backward: Callable[[], None] = lambda: None
        self.operation: Optional[str] = None

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def size(self):
        return self.tensor.size

    @property
    def runtime(self):
        return self.tensor.runtime

    @property
    def address(self):
        return self.tensor.address

    def flat(self):
        return self.tensor.flat()

    def tolist(self):
        return self.tensor.tolist()

    def item(self):
        return self.tensor.item()

    def __repr__(self):
        return (
            "AutoTensor("
            f"{self.tensor.tolist()}, "
            f"shape={self.shape}, "
            f"requires_grad={self.requires_grad}"
            ")"
        )

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = zeros_like(self.tensor)

    def _accumulate_grad(self, grad: Tensor) -> None:
        if not self.requires_grad:
            return

        if grad.shape != self.shape:
            raise ValueError(
                "Gradient shape mismatch: "
                f"{grad.shape} != {self.shape}"
            )

        if self.grad is None:
            self.grad = grad.clone()
        else:
            add_inplace(self.grad, grad)

    def backward(self, gradient: Optional[Tensor] = None) -> None:
        """Perform reverse-mode automatic differentiation."""
        if gradient is None:
            if self.size != 1:
                raise ValueError(
                    "backward() without gradient requires scalar output."
                )
            gradient = ones_like(self.tensor)

        if gradient.shape != self.shape:
            raise ValueError("Initial gradient shape mismatch.")

        topo: List[AutoTensor] = []
        visited: Set[int] = set()

        def build(node: "AutoTensor"):
            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)
            for parent in node.parents:
                build(parent)
            topo.append(node)

        build(self)
        self.grad = gradient.clone()

        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        other = as_autotensor(other, runtime=self.runtime)

        if self.shape != other.shape:
            raise ValueError("AutoTensor addition requires matching shapes.")

        result_tensor = self.tensor + other.tensor
        output = AutoTensor(
            result_tensor,
            requires_grad=(self.requires_grad or other.requires_grad),
        )
        output.parents = [self, other]
        output.operation = "add"

        def backward():
            if output.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(output.grad)
            if other.requires_grad:
                other._accumulate_grad(output.grad)

        output._backward = backward
        return output

    def __sub__(self, other):
        other = as_autotensor(other, runtime=self.runtime)

        if self.shape != other.shape:
            raise ValueError("AutoTensor subtraction requires matching shapes.")

        result_tensor = self.tensor - other.tensor
        output = AutoTensor(
            result_tensor,
            requires_grad=(self.requires_grad or other.requires_grad),
        )
        output.parents = [self, other]
        output.operation = "sub"

        def backward():
            if output.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(output.grad)
            if other.requires_grad:
                other._accumulate_grad(output.grad * -1.0)

        output._backward = backward
        return output

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            scalar = float(other)
            result_tensor = self.tensor * scalar
            output = AutoTensor(result_tensor, requires_grad=self.requires_grad)
            output.parents = [self]
            output.operation = "mul_scalar"

            def backward():
                if output.grad is None or not self.requires_grad:
                    return
                self._accumulate_grad(output.grad * scalar)

            output._backward = backward
            return output

        other = as_autotensor(other, runtime=self.runtime)

        if self.shape != other.shape:
            raise ValueError(
                "Element-wise multiplication requires matching shapes."
            )

        result_tensor = self.tensor * other.tensor
        output = AutoTensor(
            result_tensor,
            requires_grad=(self.requires_grad or other.requires_grad),
        )
        output.parents = [self, other]
        output.operation = "mul"

        def backward():
            if output.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(output.grad * other.tensor)
            if other.requires_grad:
                other._accumulate_grad(output.grad * self.tensor)

        output._backward = backward
        return output

    def __truediv__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError(
                "AutoTensor division currently supports scalar divisor only."
            )

        scalar = float(other)
        if scalar == 0.0:
            raise ZeroDivisionError

        result_tensor = self.tensor / scalar
        output = AutoTensor(result_tensor, requires_grad=self.requires_grad)
        output.parents = [self]
        output.operation = "div_scalar"

        def backward():
            if output.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(output.grad / scalar)

        output._backward = backward
        return output

    def __matmul__(self, other):
        other = as_autotensor(other, runtime=self.runtime)

        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError("matmul currently supports 2-D tensors only.")

        result_tensor = self.tensor @ other.tensor
        output = AutoTensor(
            result_tensor,
            requires_grad=(self.requires_grad or other.requires_grad),
        )
        output.parents = [self, other]
        output.operation = "matmul"

        def backward():
            if output.grad is None:
                return
            if self.requires_grad:
                self._accumulate_grad(output.grad @ other.tensor.T)
            if other.requires_grad:
                other._accumulate_grad(self.tensor.T @ output.grad)

        output._backward = backward
        return output

    @property
    def T(self):
        result_tensor = self.tensor.T
        output = AutoTensor(result_tensor, requires_grad=self.requires_grad)
        output.parents = [self]
        output.operation = "transpose"

        def backward():
            if output.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(output.grad.T)

        output._backward = backward
        return output

    def sum(self):
        value = sum(self.tensor.flat())
        result_tensor = Tensor(float(value), runtime=self.runtime)
        output = AutoTensor(result_tensor, requires_grad=self.requires_grad)
        output.parents = [self]
        output.operation = "sum"

        def backward():
            if output.grad is None or not self.requires_grad:
                return
            scalar = output.grad.item()
            self._accumulate_grad(fill_like(self.tensor, scalar))

        output._backward = backward
        return output

    def mean(self):
        return self.sum() / float(self.size)

    def exp(self):
        result_values = [math.exp(value) for value in self.flat()]
        result_tensor = Tensor(
            _reshape_values(result_values, self.shape),
            runtime=self.runtime,
        )
        output = AutoTensor(result_tensor, requires_grad=self.requires_grad)
        output.parents = [self]
        output.operation = "exp"

        def backward():
            if output.grad is None or not self.requires_grad:
                return
            self._accumulate_grad(output.grad * output.tensor)

        output._backward = backward
        return output

    def log(self):
        result_values = []
        for value in self.flat():
            if value <= 0.0:
                raise ValueError("log() requires positive values.")
            result_values.append(math.log(value))

        result_tensor = Tensor(
            _reshape_values(result_values, self.shape),
            runtime=self.runtime,
        )
        output = AutoTensor(result_tensor, requires_grad=self.requires_grad)
        output.parents = [self]
        output.operation = "log"

        def backward():
            if output.grad is None or not self.requires_grad:
                return

            upstream = output.grad.flat()
            source = self.tensor.flat()
            grad_values = [g / x for g, x in zip(upstream, source)]
            grad = Tensor(
                _reshape_values(grad_values, self.shape),
                runtime=self.runtime,
            )
            self._accumulate_grad(grad)

        output._backward = backward
        return output

    def tanh(self):
        result_values = [math.tanh(value) for value in self.flat()]
        result_tensor = Tensor(
            _reshape_values(result_values, self.shape),
            runtime=self.runtime,
        )
        output = AutoTensor(result_tensor, requires_grad=self.requires_grad)
        output.parents = [self]
        output.operation = "tanh"

        def backward():
            if output.grad is None or not self.requires_grad:
                return

            upstream = output.grad.flat()
            y_values = output.tensor.flat()
            grad_values = [
                g * (1.0 - y * y)
                for g, y in zip(upstream, y_values)
            ]
            grad = Tensor(
                _reshape_values(grad_values, self.shape),
                runtime=self.runtime,
            )
            self._accumulate_grad(grad)

        output._backward = backward
        return output


class Parameter(AutoTensor):
    """Trainable model parameter."""

    def __init__(self, tensor: Tensor, name: Optional[str] = None):
        super().__init__(tensor=tensor, requires_grad=True, name=name)


def as_autotensor(
    value,
    runtime: Optional[TensorRuntime] = None,
) -> AutoTensor:
    """Convert Tensor / scalar into AutoTensor."""
    if isinstance(value, AutoTensor):
        return value

    if isinstance(value, Tensor):
        return AutoTensor(value, requires_grad=False)

    if isinstance(value, (int, float)):
        if runtime is None:
            runtime = get_default_runtime()
        tensor = Tensor(float(value), runtime=runtime)
        return AutoTensor(tensor, requires_grad=False)

    raise TypeError("Cannot convert value to AutoTensor.")


def _reshape_values(values, shape):
    """Convert flat Python list into nested lists."""
    if len(shape) == 0:
        return values[0]

    if len(shape) == 1:
        return list(values[:shape[0]])

    stride = 1
    for dimension in shape[1:]:
        stride *= dimension

    result = []
    for i in range(shape[0]):
        start = i * stride
        end = start + stride
        result.append(_reshape_values(values[start:end], shape[1:]))

    return result


def print_graph(
    node: AutoTensor,
    indent: int = 0,
    visited=None,
) -> None:
    """Print computation graph."""
    if visited is None:
        visited = set()

    node_id = id(node)
    if node_id in visited:
        return

    visited.add(node_id)
    prefix = " " * indent
    name = node.name if node.name is not None else "unnamed"
    operation = node.operation if node.operation is not None else "leaf"

    print(prefix + f"{name}: {operation} shape={node.shape}")

    for parent in node.parents:
        print_graph(parent, indent + 2, visited)


if __name__ == "__main__":
    print()
    print("================================")
    print("Autograd Test")
    print("================================")
    print()

    x_tensor = Tensor([
        1.0,
        2.0,
        3.0,
        4.0,
    ])

    x = Parameter(x_tensor, name="x")
    y = (x * x).mean()
    y.name = "loss"

    print("x:")
    print(x)
    print()
    print("loss:")
    print(y.item())
    print()
    print("Computation graph:")
    print_graph(y)
    print()

    y.backward()

    print("Gradient dx:")
    print(x.grad)
    print()
    print("Expected:")
    print([0.5, 1.0, 1.5, 2.0])
    print()

    print("================================")
    print("Matrix Autograd Test")
    print("================================")

    a = Parameter(
        Tensor([
            [1.0, 2.0],
            [3.0, 4.0],
        ]),
        name="A",
    )

    b = Parameter(
        Tensor([
            [5.0, 6.0],
            [7.0, 8.0],
        ]),
        name="B",
    )

    c = a @ b
    loss = c.mean()

    print("C = A @ B:")
    print(c)
    print()
    print("Loss:")
    print(loss.item())
    print()

    loss.backward()

    print("dA:")
    print(a.grad)
    print()
    print("dB:")
    print(b.grad)
