# attention.py
#
# Single-head self-attention with v0.3 backward propagation.

import math
from typing import List, Optional

from tensor import Tensor, TensorRuntime, get_default_runtime
from autograd import Parameter
from layers import Linear, Softmax


class SelfAttention:
    def __init__(
        self,
        d_model: int,
        causal: bool = True,
        runtime: Optional[TensorRuntime] = None,
        seed: int = 0,
    ):
        if d_model <= 0:
            raise ValueError("d_model must be greater than 0.")

        self.d_model = d_model
        self.causal = causal
        self.runtime = runtime or get_default_runtime()

        self.q_proj = Linear(d_model, d_model, bias=False, runtime=self.runtime, seed=seed + 1)
        self.k_proj = Linear(d_model, d_model, bias=False, runtime=self.runtime, seed=seed + 2)
        self.v_proj = Linear(d_model, d_model, bias=False, runtime=self.runtime, seed=seed + 3)
        self.out_proj = Linear(d_model, d_model, bias=False, runtime=self.runtime, seed=seed + 4)
        self.softmax = Softmax()

        self._q = None
        self._k = None
        self._v = None
        self._attention_weights = None

    def forward(self, x: Tensor) -> Tensor:
        if x.runtime is not self.runtime:
            raise ValueError("Tensor runtime does not match SelfAttention runtime.")
        if len(x.shape) != 2 or x.shape[1] != self.d_model:
            raise ValueError("SelfAttention input dimension mismatch.")

        self._q = self.q_proj(x)
        self._k = self.k_proj(x)
        self._v = self.v_proj(x)

        scores = self._q @ self._k.T
        scores = scores / math.sqrt(float(self.d_model))

        if self.causal:
            scores = self._apply_causal_mask(scores)

        self._attention_weights = self.softmax(scores)
        context = self._attention_weights @ self._v
        return self.out_proj(context)

    def backward(self, grad_output: Tensor) -> Tensor:
        if any(value is None for value in (
            self._q, self._k, self._v, self._attention_weights
        )):
            raise RuntimeError("SelfAttention.backward() called before forward().")

        grad_context = self.out_proj.backward(grad_output)

        grad_attention = grad_context @ self._v.T
        grad_v_from_context = self._attention_weights.T @ grad_context

        grad_scores = self.softmax.backward(grad_attention)

        if self.causal:
            rows, cols = grad_scores.shape
            for row in range(rows):
                for col in range(cols):
                    if col > row:
                        self.runtime.memory.write_scalar(
                            grad_scores.address + row * cols + col,
                            0.0,
                        )

        scale = 1.0 / math.sqrt(float(self.d_model))
        grad_q = (grad_scores @ self._k) * scale
        grad_k = (grad_scores.T @ self._q) * scale

        grad_x_q = self.q_proj.backward(grad_q)
        grad_x_k = self.k_proj.backward(grad_k)
        grad_x_v = self.v_proj.backward(grad_v_from_context)

        return (grad_x_q + grad_x_k) + grad_x_v

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> List[Parameter]:
        result = []
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            result.extend(layer.parameters())
        return result

    def _apply_causal_mask(self, scores: Tensor) -> Tensor:
        rows, cols = scores.shape
        if rows != cols:
            raise ValueError("Causal attention score matrix must be square.")

        result = scores.clone()
        for row in range(rows):
            for col in range(cols):
                if col > row:
                    self.runtime.memory.write_scalar(
                        result.address + row * cols + col,
                        -1.0e9,
                    )
        return result

    def get_attention_weights(self, x: Tensor) -> Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        scores = (q @ k.T) / math.sqrt(float(self.d_model))
        if self.causal:
            scores = self._apply_causal_mask(scores)
        return self.softmax(scores)


if __name__ == "__main__":
    x = Tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    attention = SelfAttention(4, runtime=x.runtime, seed=42)
    y = attention(x)
    dy = Tensor.ones(y.shape, runtime=x.runtime)
    dx = attention.backward(dy)
    print("Y:", y)
    print("dX:", dx)
