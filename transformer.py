# transformer.py
#
# Transformer block/stack with v0.3 backward propagation.

from typing import List, Optional

from tensor import Tensor, TensorRuntime, get_default_runtime
from autograd import Parameter
from layers import LayerNorm
from attention import SelfAttention
from ffn import FeedForward


class TransformerBlock:
    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        causal: bool = True,
        runtime: Optional[TensorRuntime] = None,
        seed: int = 0,
    ):
        if d_model <= 0:
            raise ValueError("d_model must be greater than 0.")
        if hidden_dim is None:
            hidden_dim = 4 * d_model

        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.causal = causal
        self.runtime = runtime or get_default_runtime()

        self.norm1 = LayerNorm(d_model, runtime=self.runtime)
        self.attention = SelfAttention(
            d_model=d_model,
            causal=causal,
            runtime=self.runtime,
            seed=seed + 10,
        )
        self.norm2 = LayerNorm(d_model, runtime=self.runtime)
        self.ffn = FeedForward(
            d_model=d_model,
            hidden_dim=hidden_dim,
            runtime=self.runtime,
            seed=seed + 20,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.runtime is not self.runtime:
            raise ValueError("Tensor runtime does not match TransformerBlock runtime.")
        if len(x.shape) != 2 or x.shape[1] != self.d_model:
            raise ValueError("TransformerBlock input dimension mismatch.")

        attention_output = self.attention(self.norm1(x))
        residual1 = x + attention_output
        ffn_output = self.ffn(self.norm2(residual1))
        return residual1 + ffn_output

    def backward(self, grad_output: Tensor) -> Tensor:
        # output = residual1 + ffn(norm2(residual1))
        grad_residual1 = grad_output
        grad_ffn_input = self.ffn.backward(grad_output)
        grad_residual1_from_ffn = self.norm2.backward(grad_ffn_input)
        grad_residual1_total = grad_residual1 + grad_residual1_from_ffn

        # residual1 = x + attention(norm1(x))
        grad_x_residual = grad_residual1_total
        grad_attention_input = self.attention.backward(grad_residual1_total)
        grad_x_attention = self.norm1.backward(grad_attention_input)

        return grad_x_residual + grad_x_attention

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> List[Parameter]:
        result = []
        result.extend(self.norm1.parameters())
        result.extend(self.attention.parameters())
        result.extend(self.norm2.parameters())
        result.extend(self.ffn.parameters())
        return result


class Transformer:
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
            raise ValueError("num_layers must be greater than 0.")
        if d_model <= 0:
            raise ValueError("d_model must be greater than 0.")
        if hidden_dim is None:
            hidden_dim = 4 * d_model

        self.num_layers = num_layers
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.causal = causal
        self.runtime = runtime or get_default_runtime()

        self.blocks: List[TransformerBlock] = [
            TransformerBlock(
                d_model=d_model,
                hidden_dim=hidden_dim,
                causal=causal,
                runtime=self.runtime,
                seed=seed + layer_index * 100,
            )
            for layer_index in range(num_layers)
        ]

        self.final_norm = (
            LayerNorm(d_model, runtime=self.runtime)
            if final_norm else None
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.runtime is not self.runtime:
            raise ValueError("Tensor runtime does not match Transformer runtime.")
        if len(x.shape) != 2 or x.shape[1] != self.d_model:
            raise ValueError("Transformer input dimension mismatch.")

        for block in self.blocks:
            x = block(x)
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x

    def backward(self, grad_output: Tensor) -> Tensor:
        grad = grad_output
        if self.final_norm is not None:
            grad = self.final_norm.backward(grad)
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        return grad

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> List[Parameter]:
        result = []
        for block in self.blocks:
            result.extend(block.parameters())
        if self.final_norm is not None:
            result.extend(self.final_norm.parameters())
        return result


if __name__ == "__main__":
    x = Tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
    model = Transformer(1, 4, hidden_dim=16, runtime=x.runtime, seed=100)
    y = model(x)
    dy = Tensor.ones(y.shape, runtime=x.runtime)
    dx = model.backward(dy)
    print("Y:", y)
    print("dX:", dx)
