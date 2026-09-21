# embedding.py
#
# Embedding layer with v0.3 backward propagation.

import random
from typing import List, Optional

from tensor import Tensor, TensorRuntime, get_default_runtime
from autograd import Parameter


class Embedding:
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        runtime: Optional[TensorRuntime] = None,
        seed: int = 0,
        init_scale: float = 0.02,
    ):
        if vocab_size <= 0:
            raise ValueError("vocab_size must be greater than 0.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0.")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.runtime = runtime or get_default_runtime()
        self._last_token_ids: Optional[List[int]] = None

        rng = random.Random(seed)
        weights = [
            [rng.uniform(-init_scale, init_scale) for _ in range(embedding_dim)]
            for _ in range(vocab_size)
        ]

        self.weight = Parameter(
            Tensor(weights, runtime=self.runtime),
            name="embedding.weight",
        )

    def forward(self, token_ids: List[int]) -> Tensor:
        if not isinstance(token_ids, list) or len(token_ids) == 0:
            raise ValueError("token_ids must be a non-empty list.")

        for token_id in token_ids:
            if not isinstance(token_id, int):
                raise TypeError("Each token ID must be int.")
            if token_id < 0 or token_id >= self.vocab_size:
                raise IndexError(f"Token ID out of range: {token_id}")

        self._last_token_ids = list(token_ids)

        output = Tensor.zeros(
            (len(token_ids), self.embedding_dim),
            runtime=self.runtime,
        )

        for position, token_id in enumerate(token_ids):
            self.runtime.memory.copy(
                dst_address=output.address + position * self.embedding_dim,
                src_address=self.weight.address + token_id * self.embedding_dim,
                size=self.embedding_dim,
            )

        return output

    def backward(self, grad_output: Tensor) -> None:
        if self._last_token_ids is None:
            raise RuntimeError("Embedding.backward() called before forward().")
        expected = (len(self._last_token_ids), self.embedding_dim)
        if grad_output.shape != expected:
            raise ValueError(
                f"Embedding gradient shape mismatch: {grad_output.shape} != {expected}"
            )

        grad_values = [0.0] * (self.vocab_size * self.embedding_dim)
        upstream = grad_output.flat()

        for position, token_id in enumerate(self._last_token_ids):
            src = position * self.embedding_dim
            dst = token_id * self.embedding_dim
            for i in range(self.embedding_dim):
                grad_values[dst + i] += upstream[src + i]

        rows = [
            grad_values[row * self.embedding_dim:(row + 1) * self.embedding_dim]
            for row in range(self.vocab_size)
        ]
        self.weight.grad = Tensor(rows, runtime=self.runtime)

    def __call__(self, token_ids: List[int]) -> Tensor:
        return self.forward(token_ids)

    def parameters(self) -> List[Parameter]:
        return [self.weight]

    def get_vector(self, token_id: int) -> Tensor:
        if token_id < 0 or token_id >= self.vocab_size:
            raise IndexError(f"Token ID out of range: {token_id}")

        result = Tensor.zeros((self.embedding_dim,), runtime=self.runtime)
        self.runtime.memory.copy(
            dst_address=result.address,
            src_address=self.weight.address + token_id * self.embedding_dim,
            size=self.embedding_dim,
        )
        return result

    def set_vector(self, token_id: int, values: List[float]) -> None:
        if token_id < 0 or token_id >= self.vocab_size:
            raise IndexError(f"Token ID out of range: {token_id}")
        if len(values) != self.embedding_dim:
            raise ValueError("Embedding vector size mismatch.")

        self.runtime.memory.write(
            self.weight.address + token_id * self.embedding_dim,
            values,
        )


if __name__ == "__main__":
    embedding = Embedding(vocab_size=12, embedding_dim=4, seed=42)
    output = embedding([1, 3, 5, 3, 2])
    grad = Tensor.ones(output.shape, runtime=output.runtime)
    embedding.backward(grad)
    print("Output:", output)
    print("Gradient shape:", embedding.weight.grad.shape)
