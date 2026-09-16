# embedding.py
#
# Simple Embedding Layer using trainable Parameter weights.

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
        if init_scale <= 0:
            raise ValueError("init_scale must be greater than 0.")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.runtime = runtime or get_default_runtime()

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
        if not isinstance(token_ids, list):
            raise TypeError("token_ids must be a list.")
        if len(token_ids) == 0:
            raise ValueError("token_ids must not be empty.")

        for token_id in token_ids:
            if not isinstance(token_id, int):
                raise TypeError("Each token ID must be int.")
            if token_id < 0 or token_id >= self.vocab_size:
                raise IndexError(f"Token ID out of range: {token_id}")

        output = Tensor.zeros(
            (len(token_ids), self.embedding_dim),
            runtime=self.runtime,
        )

        for position, token_id in enumerate(token_ids):
            src_address = self.weight.address + token_id * self.embedding_dim
            dst_address = output.address + position * self.embedding_dim
            self.runtime.memory.copy(
                dst_address=dst_address,
                src_address=src_address,
                size=self.embedding_dim,
            )

        return output

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
            raise ValueError(
                f"Embedding vector size mismatch: expected {self.embedding_dim}, "
                f"got {len(values)}"
            )

        self.runtime.memory.write(
            self.weight.address + token_id * self.embedding_dim,
            values,
        )

    def info(self) -> None:
        print("Embedding Layer")
        print("===============")
        print(f"Vocabulary size : {self.vocab_size}")
        print(f"Embedding dim   : {self.embedding_dim}")
        print(f"Weight shape    : {self.weight.shape}")
        print(f"GPU address     : {self.weight.address}")


if __name__ == "__main__":
    embedding = Embedding(vocab_size=12, embedding_dim=4, seed=42)
    output = embedding([1, 3, 5, 3, 2])
    print(output)
    print("Parameters:", len(embedding.parameters()))
