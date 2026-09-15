# embedding.py
#
# Simple Embedding Layer
#
# Purpose:
#   - Convert token IDs into embedding vectors
#   - Store embedding weights using Tensor
#   - Return a Tensor of shape:
#
#       [sequence_length, embedding_dim]
#
# Dependencies:
#   tensor.py
#
# Notes:
#   - No NumPy is used.
#   - Weight initialization is intentionally simple.
#   - Backpropagation is not implemented yet.

import random
from typing import List, Optional

from tensor import Tensor, TensorRuntime, get_default_runtime


class Embedding:
    """
    Simple embedding layer.

    Parameters
    ----------
    vocab_size:
        Number of tokens in vocabulary.

    embedding_dim:
        Size of each embedding vector.

    runtime:
        Optional TensorRuntime.

    seed:
        Random seed for reproducibility.

    init_scale:
        Initial random weight scale.

    Example
    -------

    embedding = Embedding(
        vocab_size=100,
        embedding_dim=16
    )

    token_ids = [1, 5, 8, 3]

    x = embedding(token_ids)

    print(x.shape)

    # (4, 16)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        runtime: Optional[TensorRuntime] = None,
        seed: int = 0,
        init_scale: float = 0.02,
    ):

        if vocab_size <= 0:
            raise ValueError(
                "vocab_size must be greater than 0."
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than 0."
            )

        if init_scale <= 0:
            raise ValueError(
                "init_scale must be greater than 0."
            )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        if runtime is None:
            runtime = get_default_runtime()

        self.runtime = runtime

        random.seed(seed)

        # ----------------------------------------------------
        # Initialize embedding matrix
        #
        # Shape:
        #   [vocab_size, embedding_dim]
        # ----------------------------------------------------

        weights = []

        for _ in range(vocab_size):

            row = []

            for _ in range(embedding_dim):

                value = random.uniform(
                    -init_scale,
                    init_scale
                )

                row.append(
                    value
                )

            weights.append(
                row
            )

        self.weight = Tensor(
            weights,
            runtime=self.runtime
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        token_ids: List[int]
    ) -> Tensor:
        """
        Convert token IDs into embedding vectors.

        Input:
            token_ids:
                [t0, t1, t2, ...]

        Output:
            Tensor shape:
                [sequence_length, embedding_dim]
        """

        if not isinstance(
            token_ids,
            list
        ):
            raise TypeError(
                "token_ids must be a list."
            )

        sequence_length = len(
            token_ids
        )

        if sequence_length == 0:

            return Tensor.zeros(
                (
                    0,
                    self.embedding_dim
                ),
                runtime=self.runtime
            )

        # ----------------------------------------------------
        # Validate token IDs
        # ----------------------------------------------------

        for token_id in token_ids:

            if not isinstance(
                token_id,
                int
            ):
                raise TypeError(
                    "Each token ID must be int."
                )

            if (
                token_id < 0
                or token_id >= self.vocab_size
            ):
                raise IndexError(
                    f"Token ID out of range: "
                    f"{token_id}"
                )

        # ----------------------------------------------------
        # Allocate output Tensor
        #
        # Shape:
        #   [sequence_length, embedding_dim]
        # ----------------------------------------------------

        output = Tensor.zeros(
            (
                sequence_length,
                self.embedding_dim
            ),
            runtime=self.runtime
        )

        # ----------------------------------------------------
        # Copy corresponding rows
        # ----------------------------------------------------

        for position, token_id in enumerate(
            token_ids
        ):

            src_address = (
                self.weight.address
                + token_id
                * self.embedding_dim
            )

            dst_address = (
                output.address
                + position
                * self.embedding_dim
            )

            self.runtime.memory.copy(
                dst_address=dst_address,
                src_address=src_address,
                size=self.embedding_dim
            )

        return output

    # ========================================================
    # Callable Interface
    # ========================================================

    def __call__(
        self,
        token_ids: List[int]
    ) -> Tensor:

        return self.forward(
            token_ids
        )

    # ========================================================
    # Access One Embedding Vector
    # ========================================================

    def get_vector(
        self,
        token_id: int
    ) -> Tensor:
        """
        Return one token embedding vector.

        Shape:
            [embedding_dim]
        """

        if (
            token_id < 0
            or token_id >= self.vocab_size
        ):
            raise IndexError(
                f"Token ID out of range: "
                f"{token_id}"
            )

        result = Tensor.zeros(
            (
                self.embedding_dim,
            ),
            runtime=self.runtime
        )

        src_address = (
            self.weight.address
            + token_id
            * self.embedding_dim
        )

        self.runtime.memory.copy(
            dst_address=result.address,
            src_address=src_address,
            size=self.embedding_dim
        )

        return result

    # ========================================================
    # Set One Embedding Vector
    # ========================================================

    def set_vector(
        self,
        token_id: int,
        values: List[float]
    ) -> None:
        """
        Replace one embedding vector.
        """

        if (
            token_id < 0
            or token_id >= self.vocab_size
        ):
            raise IndexError(
                f"Token ID out of range: "
                f"{token_id}"
            )

        if len(values) != self.embedding_dim:

            raise ValueError(
                "Embedding vector size mismatch: "
                f"expected {self.embedding_dim}, "
                f"got {len(values)}"
            )

        address = (
            self.weight.address
            + token_id
            * self.embedding_dim
        )

        self.runtime.memory.write(
            address,
            values
        )

    # ========================================================
    # Information
    # ========================================================

    def info(self) -> None:
        """
        Print embedding layer information.
        """

        print(
            "Embedding Layer"
        )

        print(
            "==============="
        )

        print(
            f"Vocabulary size : "
            f"{self.vocab_size}"
        )

        print(
            f"Embedding dim   : "
            f"{self.embedding_dim}"
        )

        print(
            f"Weight shape    : "
            f"{self.weight.shape}"
        )

        print(
            f"GPU address     : "
            f"{self.weight.address}"
        )


# ============================================================
# Simple Test
# ============================================================

if __name__ == "__main__":

    print()

    print(
        "================================"
    )

    print(
        "Embedding Test"
    )

    print(
        "================================"
    )

    print()

    # --------------------------------------------------------
    # Example vocabulary
    # --------------------------------------------------------

    vocab_size = 12
    embedding_dim = 4

    embedding = Embedding(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        seed=42
    )

    embedding.info()

    print()

    # --------------------------------------------------------
    # One token
    # --------------------------------------------------------

    token_id = 3

    vector = embedding.get_vector(
        token_id
    )

    print(
        f"Token {token_id} vector:"
    )

    print(
        vector
    )

    print()

    # --------------------------------------------------------
    # Sequence
    # --------------------------------------------------------

    token_ids = [
        1,
        3,
        5,
        3,
        2
    ]

    output = embedding(
        token_ids
    )

    print(
        "Token IDs:"
    )

    print(
        token_ids
    )

    print()

    print(
        "Embedding output:"
    )

    print(
        output
    )

    print()

    print(
        "Output shape:"
    )

    print(
        output.shape
    )

    print()

    # --------------------------------------------------------
    # Modify one token embedding
    # --------------------------------------------------------

    embedding.set_vector(
        token_id=3,
        values=[
            1.0,
            2.0,
            3.0,
            4.0
        ]
    )

    print(
        "Modified token 3:"
    )

    print(
        embedding.get_vector(3)
    )

    print()

    # --------------------------------------------------------
    # GPU memory information
    # --------------------------------------------------------

    embedding.runtime.memory.info()
