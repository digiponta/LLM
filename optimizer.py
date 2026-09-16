# optimizer.py
#
# Optimizers for the homemade LLM.
#
# Dependencies:
#   autograd.py
#   tensor.py
#
# Purpose:
#   - SGD
#   - Adam
#   - zero_grad()
#   - step()
#
# NumPy is intentionally not used.

import math
from typing import List, Iterable

from autograd import Parameter


# ============================================================
# Base Optimizer
# ============================================================

class Optimizer:
    """
    Base optimizer class.
    """

    def __init__(
        self,
        parameters: Iterable[Parameter]
    ):

        self.parameters: List[Parameter] = list(
            parameters
        )

        if len(self.parameters) == 0:

            raise ValueError(
                "Optimizer received no parameters."
            )

        for parameter in self.parameters:

            if not isinstance(
                parameter,
                Parameter
            ):

                raise TypeError(
                    "Optimizer parameters must "
                    "be Parameter objects."
                )

    # ========================================================
    # Zero Gradients
    # ========================================================

    def zero_grad(
        self
    ) -> None:

        for parameter in self.parameters:

            parameter.grad = None

    # ========================================================
    # Step
    # ========================================================

    def step(
        self
    ) -> None:

        raise NotImplementedError


# ============================================================
# SGD
# ============================================================

class SGD(Optimizer):
    """
    Stochastic Gradient Descent.

    Update rule:

        parameter =
            parameter
            - learning_rate * gradient

    Optional momentum:

        velocity =
            momentum * velocity
            + gradient

        parameter =
            parameter
            - learning_rate * velocity
    """

    def __init__(
        self,
        parameters: Iterable[Parameter],
        learning_rate: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):

        super().__init__(
            parameters
        )

        if learning_rate <= 0.0:

            raise ValueError(
                "learning_rate must be > 0."
            )

        if momentum < 0.0:

            raise ValueError(
                "momentum must be >= 0."
            )

        if weight_decay < 0.0:

            raise ValueError(
                "weight_decay must be >= 0."
            )

        self.learning_rate = float(
            learning_rate
        )

        self.momentum = float(
            momentum
        )

        self.weight_decay = float(
            weight_decay
        )

        # ----------------------------------------------------
        # Momentum buffers
        # ----------------------------------------------------

        self.velocity = []

        for parameter in self.parameters:

            self.velocity.append(
                [0.0] * parameter.size
            )

    # ========================================================
    # Step
    # ========================================================

    def step(
        self
    ) -> None:

        for parameter_index, parameter in enumerate(
            self.parameters
        ):

            if parameter.grad is None:
                continue

            gradient_values = (
                parameter.grad.flat()
            )

            parameter_values = (
                parameter.tensor.flat()
            )

            velocity = self.velocity[
                parameter_index
            ]

            for i in range(
                parameter.size
            ):

                gradient = gradient_values[
                    i
                ]

                # --------------------------------------------
                # Weight decay
                # --------------------------------------------

                if self.weight_decay != 0.0:

                    gradient += (
                        self.weight_decay
                        * parameter_values[i]
                    )

                # --------------------------------------------
                # Momentum
                # --------------------------------------------

                if self.momentum != 0.0:

                    velocity[i] = (
                        self.momentum
                        * velocity[i]
                        + gradient
                    )

                    update = velocity[
                        i
                    ]

                else:

                    update = gradient

                # --------------------------------------------
                # Parameter update
                # --------------------------------------------

                new_value = (
                    parameter_values[i]
                    - self.learning_rate
                    * update
                )

                parameter.runtime.memory.write_scalar(
                    parameter.address + i,
                    new_value
                )


# ============================================================
# Adam
# ============================================================

class Adam(Optimizer):
    """
    Adam optimizer.

    Update rule:

        m_t =
            beta1 * m_(t-1)
            + (1-beta1) * grad

        v_t =
            beta2 * v_(t-1)
            + (1-beta2) * grad^2

        m_hat =
            m_t / (1-beta1^t)

        v_hat =
            v_t / (1-beta2^t)

        parameter =
            parameter
            - lr * m_hat
              / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        parameters: Iterable[Parameter],
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):

        super().__init__(
            parameters
        )

        if learning_rate <= 0.0:

            raise ValueError(
                "learning_rate must be > 0."
            )

        if not (
            0.0 <= beta1 < 1.0
        ):

            raise ValueError(
                "beta1 must satisfy "
                "0 <= beta1 < 1."
            )

        if not (
            0.0 <= beta2 < 1.0
        ):

            raise ValueError(
                "beta2 must satisfy "
                "0 <= beta2 < 1."
            )

        if eps <= 0.0:

            raise ValueError(
                "eps must be > 0."
            )

        if weight_decay < 0.0:

            raise ValueError(
                "weight_decay must be >= 0."
            )

        self.learning_rate = float(
            learning_rate
        )

        self.beta1 = float(
            beta1
        )

        self.beta2 = float(
            beta2
        )

        self.eps = float(
            eps
        )

        self.weight_decay = float(
            weight_decay
        )

        self.step_count = 0

        # ----------------------------------------------------
        # Adam state
        # ----------------------------------------------------

        self.m = []
        self.v = []

        for parameter in self.parameters:

            self.m.append(
                [0.0] * parameter.size
            )

            self.v.append(
                [0.0] * parameter.size
            )

    # ========================================================
    # Step
    # ========================================================

    def step(
        self
    ) -> None:

        self.step_count += 1

        beta1_power = (
            self.beta1
            ** self.step_count
        )

        beta2_power = (
            self.beta2
            ** self.step_count
        )

        for parameter_index, parameter in enumerate(
            self.parameters
        ):

            if parameter.grad is None:
                continue

            gradient_values = (
                parameter.grad.flat()
            )

            parameter_values = (
                parameter.tensor.flat()
            )

            m_state = self.m[
                parameter_index
            ]

            v_state = self.v[
                parameter_index
            ]

            for i in range(
                parameter.size
            ):

                gradient = gradient_values[
                    i
                ]

                # --------------------------------------------
                # Weight decay
                # --------------------------------------------

                if self.weight_decay != 0.0:

                    gradient += (
                        self.weight_decay
                        * parameter_values[i]
                    )

                # --------------------------------------------
                # First moment
                # --------------------------------------------

                m_state[i] = (
                    self.beta1
                    * m_state[i]
                    + (
                        1.0
                        - self.beta1
                    )
                    * gradient
                )

                # --------------------------------------------
                # Second moment
                # --------------------------------------------

                v_state[i] = (
                    self.beta2
                    * v_state[i]
                    + (
                        1.0
                        - self.beta2
                    )
                    * gradient
                    * gradient
                )

                # --------------------------------------------
                # Bias correction
                # --------------------------------------------

                m_hat = (
                    m_state[i]
                    / (
                        1.0
                        - beta1_power
                    )
                )

                v_hat = (
                    v_state[i]
                    / (
                        1.0
                        - beta2_power
                    )
                )

                # --------------------------------------------
                # Parameter update
                # --------------------------------------------

                update = (
                    self.learning_rate
                    * m_hat
                    / (
                        math.sqrt(
                            v_hat
                        )
                        + self.eps
                    )
                )

                new_value = (
                    parameter_values[i]
                    - update
                )

                parameter.runtime.memory.write_scalar(
                    parameter.address + i,
                    new_value
                )


# ============================================================
# Simple Test
# ============================================================

if __name__ == "__main__":

    from tensor import Tensor
    from autograd import Parameter

    print()

    print(
        "================================"
    )

    print(
        "Optimizer Test"
    )

    print(
        "================================"
    )

    print()

    # --------------------------------------------------------
    # SGD Test
    #
    # Minimize:
    #
    #     loss = mean(x * x)
    #
    # Starting:
    #     x = [1, 2, 3, 4]
    #
    # Gradient:
    #     dx = 2x / 4
    # --------------------------------------------------------

    x = Parameter(
        Tensor([
            1.0,
            2.0,
            3.0,
            4.0,
        ]),
        name="x"
    )

    optimizer = SGD(
        [x],
        learning_rate=0.1
    )

    print(
        "Initial x:"
    )

    print(
        x.tensor
    )

    print()

    for step in range(
        5
    ):

        optimizer.zero_grad()

        loss = (
            x * x
        ).mean()

        loss.backward()

        optimizer.step()

        print(
            f"Step {step + 1}"
        )

        print(
            "Loss:",
            loss.item()
        )

        print(
            "x:",
            x.tensor.tolist()
        )

        print()

    # --------------------------------------------------------
    # Adam Test
    # --------------------------------------------------------

    print(
        "================================"
    )

    print(
        "Adam Test"
    )

    print(
        "================================"
    )

    y = Parameter(
        Tensor([
            3.0,
            -2.0,
            1.0,
        ]),
        name="y"
    )

    adam = Adam(
        [y],
        learning_rate=0.05
    )

    print()

    print(
        "Initial y:"
    )

    print(
        y.tensor
    )

    print()

    for step in range(
        5
    ):

        adam.zero_grad()

        loss = (
            y * y
        ).mean()

        loss.backward()

        adam.step()

        print(
            f"Step {step + 1}"
        )

        print(
            "Loss:",
            loss.item()
        )

        print(
            "y:",
            y.tensor.tolist()
        )

        print()
