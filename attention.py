# attention.py
#
# Single-head self-attention.

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

        self.q_proj = Linear(
            d_model, d_model, bias=False, runtime=self.runtime, seed=seed + 1
        )
        self.k_proj = Linear(
            d_model, d_model, bias=False, runtime=self.runtime, seed=seed + 2
        )
        self.v_proj = Linear(
            d_model, d_model, bias=False, runtime=self.runtime, seed=seed + 3
        )
        self.out_proj = Linear(
            d_model, d_model, bias=False, runtime=self.runtime, seed=seed + 4
        )
        self.softmax = Softmax()

    def forward(self, x: Tensor) -> Tensor:
        if x.runtime is not self.runtime:
            raise ValueError("Tensor runtime does not match SelfAttention runtime.")
        if len(x.shape) != 2:
            raise ValueError("SelfAttention currently expects a 2-D Tensor.")
        if x.shape[1] != self.d_model:
            raise ValueError(
                f"SelfAttention dimension mismatch: expected {self.d_model}, "
                f"got {x.shape[1]}"
            )

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = q @ k.T
        scores = scores / math.sqrt(float(self.d_model))

        if self.causal:
            scores = self._apply_causal_mask(scores)

        attention_weights = self.softmax(scores)
        context = attention_weights @ v
        return self.out_proj(context)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> List[Parameter]:
        result = []
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            result.extend(layer.parameters())
        return result

    def _apply_causal_mask(self, scores: Tensor) -> Tensor:
        if len(scores.shape) != 2:
            raise ValueError("Causal mask requires 2-D scores.")

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

    def info(self) -> None:
        print("Self Attention")
        print("==============")
        print(f"d_model    : {self.d_model}")
        print(f"Causal     : {self.causal}")
        print(f"Parameters : {len(self.parameters())}")


if __name__ == "__main__":
    x = Tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    attention = SelfAttention(4, runtime=x.runtime, seed=42)
    print(attention(x))
    attention.info()
