# loss.py
#
# Loss functions for the homemade LLM.
#
# Purpose:
#   - Cross Entropy Loss
#   - Negative Log Likelihood
#
# Dependencies:
#   tensor.py
#
# Notes:
#   - No NumPy is used.
#   - Computes scalar cross-entropy loss.
#   - Provides the analytic gradient with respect to logits for v0.2 training.

import math
from typing import List

from tensor import Tensor


# ============================================================
# Cross Entropy Loss
# ============================================================

class CrossEntropyLoss:
    """
    Cross entropy loss for language modeling.

    Input:
        logits:
            Tensor of shape

                [sequence_length, vocab_size]

        targets:
            List[int]

            One correct token ID for each sequence position.

    Output:
        Python float scalar.

    Formula
    -------

        loss_i =
            -log(
                exp(logit[target])
                /
                sum(exp(logits))
            )

    Numerically stable form:

        log_sum_exp =
            max_logit
            +
            log(
                sum(
                    exp(logit - max_logit)
                )
            )

        loss_i =
            log_sum_exp
            - target_logit
    """

    def __init__(
        self,
        reduction: str = "mean"
    ):

        if reduction not in (
            "mean",
            "sum",
            "none",
        ):

            raise ValueError(
                "reduction must be "
                "'mean', 'sum', or 'none'."
            )

        self.reduction = reduction

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        logits: Tensor,
        targets: List[int]
    ):
        """
        Compute cross entropy loss.

        Parameters
        ----------
        logits:
            Tensor shape:

                [sequence_length, vocab_size]

        targets:
            Correct token IDs.

        Returns
        -------
        reduction == "mean":
            float

        reduction == "sum":
            float

        reduction == "none":
            List[float]
        """

        if len(logits.shape) != 2:

            raise ValueError(
                "CrossEntropyLoss expects "
                "2-D logits."
            )

        sequence_length = (
            logits.shape[0]
        )

        vocab_size = (
            logits.shape[1]
        )

        if len(targets) != sequence_length:

            raise ValueError(
                "Target length mismatch: "
                f"expected {sequence_length}, "
                f"got {len(targets)}"
            )

        losses = []

        # ----------------------------------------------------
        # Process each sequence position
        # ----------------------------------------------------

        for position in range(
            sequence_length
        ):

            target_id = targets[
                position
            ]

            if not isinstance(
                target_id,
                int
            ):

                raise TypeError(
                    "Target token IDs "
                    "must be integers."
                )

            if (
                target_id < 0
                or target_id >= vocab_size
            ):

                raise IndexError(
                    "Target token ID "
                    f"out of range: "
                    f"{target_id}"
                )

            # ------------------------------------------------
            # Read one row of logits
            # ------------------------------------------------

            row_address = (
                logits.address
                + position
                * vocab_size
            )

            row = (
                logits.runtime.memory.read(
                    row_address,
                    vocab_size
                )
            )

            # ------------------------------------------------
            # Numerically stable log-sum-exp
            # ------------------------------------------------

            max_logit = max(
                row
            )

            exp_sum = 0.0

            for value in row:

                exp_sum += math.exp(
                    value - max_logit
                )

            log_sum_exp = (
                max_logit
                + math.log(
                    exp_sum
                )
            )

            # ------------------------------------------------
            # Negative log likelihood
            # ------------------------------------------------

            target_logit = row[
                target_id
            ]

            loss = (
                log_sum_exp
                - target_logit
            )

            losses.append(
                loss
            )

        # ----------------------------------------------------
        # Reduction
        # ----------------------------------------------------

        if self.reduction == "none":

            return losses

        total = sum(
            losses
        )

        if self.reduction == "sum":

            return total

        return (
            total
            / len(losses)
        )

    # ========================================================
    # Gradient with respect to logits
    # ========================================================

    def gradient(
        self,
        logits: Tensor,
        targets: List[int],
    ) -> Tensor:
        """Return d(loss)/d(logits).

        For cross entropy with softmax:

            dL/dz = softmax(z) - one_hot(target)

        Mean reduction additionally divides by sequence length.
        """

        if self.reduction == "none":
            raise ValueError(
                "gradient() does not support reduction='none'."
            )

        if len(logits.shape) != 2:
            raise ValueError(
                "CrossEntropyLoss expects 2-D logits."
            )

        sequence_length, vocab_size = logits.shape

        if len(targets) != sequence_length:
            raise ValueError(
                "Target length mismatch: "
                f"expected {sequence_length}, got {len(targets)}"
            )

        gradient_values = []

        for position in range(sequence_length):
            target_id = targets[position]

            if not isinstance(target_id, int):
                raise TypeError(
                    "Target token IDs must be integers."
                )

            if target_id < 0 or target_id >= vocab_size:
                raise IndexError(
                    f"Target token ID out of range: {target_id}"
                )

            row_address = (
                logits.address
                + position * vocab_size
            )

            row = logits.runtime.memory.read(
                row_address,
                vocab_size,
            )

            max_logit = max(row)
            exp_values = [
                math.exp(value - max_logit)
                for value in row
            ]
            total = sum(exp_values)

            if total == 0.0:
                raise ZeroDivisionError(
                    "Softmax normalization sum is zero."
                )

            row_gradient = [
                value / total
                for value in exp_values
            ]

            row_gradient[target_id] -= 1.0

            if self.reduction == "mean":
                scale = 1.0 / sequence_length
                row_gradient = [
                    value * scale
                    for value in row_gradient
                ]

            gradient_values.append(row_gradient)

        return Tensor(
            gradient_values,
            runtime=logits.runtime,
        )

    def forward_and_gradient(
        self,
        logits: Tensor,
        targets: List[int],
    ):
        """Return (loss, dloss_dlogits) for training."""
        loss = self.forward(logits, targets)
        gradient = self.gradient(logits, targets)
        return loss, gradient

    # ========================================================
    # Callable Interface
    # ========================================================

    def __call__(
        self,
        logits: Tensor,
        targets: List[int]
    ):

        return self.forward(
            logits,
            targets
        )


# ============================================================
# Language Model Loss
# ============================================================

class LanguageModelLoss:
    """
    Language-model next-token loss.

    Given tokens:

        [A, B, C, D]

    input tokens become:

        [A, B, C]

    target tokens become:

        [B, C, D]

    This class expects logits corresponding to
    the input token positions.

    Example
    -------

    token_ids = [
        1,
        5,
        8,
        2
    ]

    inputs:

        [1, 5, 8]

    targets:

        [5, 8, 2]
    """

    def __init__(
        self,
        reduction: str = "mean"
    ):

        self.cross_entropy = (
            CrossEntropyLoss(
                reduction=reduction
            )
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        logits: Tensor,
        target_ids: List[int]
    ):

        return self.cross_entropy(
            logits,
            target_ids
        )

    def __call__(
        self,
        logits: Tensor,
        target_ids: List[int]
    ):

        return self.forward(
            logits,
            target_ids
        )


# ============================================================
# Utility
# ============================================================

def shift_tokens(
    token_ids: List[int]
):
    """
    Convert a token sequence into
    language-model input / target pairs.

    Example
    -------

    input:

        [1, 4, 7, 2]

    output:

        inputs:
            [1, 4, 7]

        targets:
            [4, 7, 2]
    """

    if len(token_ids) < 2:

        raise ValueError(
            "At least two tokens are required "
            "for language-model training."
        )

    inputs = token_ids[
        :-1
    ]

    targets = token_ids[
        1:
    ]

    return (
        inputs,
        targets
    )


# ============================================================
# Perplexity
# ============================================================

def perplexity(
    loss: float
) -> float:
    """
    Convert average cross-entropy loss
    to perplexity.

        perplexity = exp(loss)
    """

    return math.exp(
        loss
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
        "Loss Test"
    )

    print(
        "================================"
    )

    print()

    # --------------------------------------------------------
    # Example logits
    #
    # sequence length = 3
    # vocab size      = 4
    # --------------------------------------------------------

    logits = Tensor([
        [
            2.0,
            1.0,
            0.0,
            -1.0,
        ],
        [
            0.0,
            3.0,
            1.0,
            0.0,
        ],
        [
            -1.0,
            0.0,
            1.0,
            4.0,
        ],
    ])

    targets = [
        0,
        1,
        3,
    ]

    print(
        "Logits:"
    )

    print(
        logits
    )

    print()

    print(
        "Targets:"
    )

    print(
        targets
    )

    print()

    # --------------------------------------------------------
    # Mean loss
    # --------------------------------------------------------

    loss_fn = CrossEntropyLoss(
        reduction="mean"
    )

    loss = loss_fn(
        logits,
        targets
    )

    print(
        "Cross entropy loss:"
    )

    print(
        loss
    )

    print()

    # --------------------------------------------------------
    # Per-position loss
    # --------------------------------------------------------

    loss_none = CrossEntropyLoss(
        reduction="none"
    )

    losses = loss_none(
        logits,
        targets
    )

    print(
        "Per-token losses:"
    )

    print(
        losses
    )

    print()

    # --------------------------------------------------------
    # Perplexity
    # --------------------------------------------------------

    ppl = perplexity(
        loss
    )

    print(
        "Perplexity:"
    )

    print(
        ppl
    )

    print()

    # --------------------------------------------------------
    # Token shifting
    # --------------------------------------------------------

    token_ids = [
        1,
        4,
        7,
        2,
    ]

    inputs, targets = shift_tokens(
        token_ids
    )

    print(
        "Original tokens:"
    )

    print(
        token_ids
    )

    print()

    print(
        "LM inputs:"
    )

    print(
        inputs
    )

    print()

    print(
        "LM targets:"
    )

    print(
        targets
    )
