# transformer.py
#
# Transformer block and stack.

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
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than 0.")

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
        if len(x.shape) != 2:
            raise ValueError("TransformerBlock currently expects a 2-D Tensor.")
        if x.shape[1] != self.d_model:
            raise ValueError(
                f"TransformerBlock dimension mismatch: expected {self.d_model}, "
                f"got {x.shape[1]}"
            )

        attention_output = self.attention(self.norm1(x))
        x = x + attention_output
        ffn_output = self.ffn(self.norm2(x))
        return x + ffn_output

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> List[Parameter]:
        result = []
        result.extend(self.norm1.parameters())
        result.extend(self.attention.parameters())
        result.extend(self.norm2.parameters())
        result.extend(self.ffn.parameters())
        return result

    def info(self) -> None:
        print("Transformer Block")
        print("=================")
        print(f"d_model    : {self.d_model}")
        print(f"hidden_dim : {self.hidden_dim}")
        print(f"causal     : {self.causal}")
        print(f"Parameters : {len(self.parameters())}")


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
            if final_norm
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.runtime is not self.runtime:
            raise ValueError("Tensor runtime does not match Transformer runtime.")
        if len(x.shape) != 2:
            raise ValueError("Transformer expects a 2-D Tensor.")
        if x.shape[1] != self.d_model:
            raise ValueError("Transformer input dimension mismatch.")

        for block in self.blocks:
            x = block(x)
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> List[Parameter]:
        result = []
        for block in self.blocks:
            result.extend(block.parameters())
        if self.final_norm is not None:
            result.extend(self.final_norm.parameters())
        return result

    def info(self) -> None:
        print("Transformer")
        print("===========")
        print(f"Layers     : {self.num_layers}")
        print(f"d_model    : {self.d_model}")
        print(f"hidden_dim : {self.hidden_dim}")
        print(f"causal     : {self.causal}")
        print(f"final norm : {self.final_norm is not None}")
        print(f"Parameters : {len(self.parameters())}")


if __name__ == "__main__":
    x = Tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    model = Transformer(2, 8, hidden_dim=32, runtime=x.runtime, seed=100)
    print(model(x))
    model.info()
