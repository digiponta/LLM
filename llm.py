# llm.py
#
# Minimal Transformer language model.

import json
import math
import time
from pathlib import Path
from typing import List, Optional

from tensor import Tensor, TensorRuntime, get_default_runtime
from autograd import Parameter
from embedding import Embedding
from transformer import Transformer
from layers import Linear, Softmax


class LanguageModel:
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        hidden_dim: Optional[int] = None,
        causal: bool = True,
        runtime: Optional[TensorRuntime] = None,
        seed: int = 0,
    ):
        if vocab_size <= 0:
            raise ValueError("vocab_size must be greater than 0.")
        if d_model <= 0:
            raise ValueError("d_model must be greater than 0.")
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than 0.")
        if hidden_dim is None:
            hidden_dim = 4 * d_model

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.causal = causal
        self.runtime = runtime or get_default_runtime()

        self.embedding = Embedding(
            vocab_size=vocab_size,
            embedding_dim=d_model,
            runtime=self.runtime,
            seed=seed + 1,
        )
        self.transformer = Transformer(
            num_layers=num_layers,
            d_model=d_model,
            hidden_dim=hidden_dim,
            causal=causal,
            runtime=self.runtime,
            seed=seed + 100,
        )
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            bias=False,
            runtime=self.runtime,
            seed=seed + 1000,
        )
        self.softmax = Softmax()

    def hidden_states(self, token_ids: List[int]) -> Tensor:
        """Return Transformer hidden states before the LM output projection."""
        if not isinstance(token_ids, list):
            raise TypeError("token_ids must be a list.")
        if len(token_ids) == 0:
            raise ValueError("token_ids must not be empty.")

        x = self.embedding(token_ids)
        return self.transformer(x)

    def forward(self, token_ids: List[int]) -> Tensor:
        x = self.hidden_states(token_ids)
        return self.lm_head(x)

    def __call__(self, token_ids: List[int]) -> Tensor:
        return self.forward(token_ids)

    def parameters(self) -> List[Parameter]:
        """Return every trainable Parameter in optimizer-ready order."""
        result = []
        result.extend(self.embedding.parameters())
        result.extend(self.transformer.parameters())
        result.extend(self.lm_head.parameters())
        return result

    def zero_grad(self) -> None:
        """Clear gradients on all model parameters."""
        for parameter in self.parameters():
            parameter.grad = None

    def probabilities(self, token_ids: List[int]) -> Tensor:
        return self.softmax(self.forward(token_ids))

    def last_logits(self, token_ids: List[int]) -> List[float]:
        logits = self.forward(token_ids)
        start_address = (
            logits.address
            + (logits.shape[0] - 1) * logits.shape[1]
        )
        return self.runtime.memory.read(start_address, logits.shape[1])

    def last_probabilities(self, token_ids: List[int]) -> List[float]:
        logits = self.last_logits(token_ids)
        max_value = max(logits)
        exp_values = [math.exp(value - max_value) for value in logits]
        total = sum(exp_values)
        return [value / total for value in exp_values]

    def next_token(self, token_ids: List[int]) -> int:
        logits = self.last_logits(token_ids)
        best_index = 0
        for index in range(1, len(logits)):
            if logits[index] > logits[best_index]:
                best_index = index
        return best_index

    def generate(
        self,
        token_ids: List[int],
        max_new_tokens: int = 10,
        eos_id: Optional[int] = None,
        show_progress: bool = False,
    ) -> List[int]:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0.")

        generated = list(token_ids)

        # Model parameters are already resident in virtual GPU memory.
        # Every next-token forward pass allocates temporary tensors after
        # this mark. Rewind those allocations after each generated token so
        # autoregressive generation does not exhaust the linear allocator.
        memory = self.runtime.memory
        temporary_mark = memory.next_address

        start_time = time.perf_counter()

        for step in range(1, max_new_tokens + 1):
            try:
                next_id = self.next_token(generated)
            finally:
                temporary_addresses = [
                    address
                    for address in memory.allocations
                    if address >= temporary_mark
                ]

                for address in temporary_addresses:
                    del memory.allocations[address]

                memory.next_address = temporary_mark

            generated.append(next_id)

            if show_progress:
                elapsed = time.perf_counter() - start_time
                tokens_per_second = step / elapsed if elapsed > 0 else 0.0
                remaining = max_new_tokens - step
                eta = (
                    remaining / tokens_per_second
                    if tokens_per_second > 0
                    else float("inf")
                )

                def format_duration(seconds: float) -> str:
                    if seconds < 0 or seconds == float("inf"):
                        return "--:--:--"
                    total_seconds = int(seconds)
                    hours, remainder = divmod(total_seconds, 3600)
                    minutes, secs = divmod(remainder, 60)
                    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

                progress = 100.0 * step / max_new_tokens if max_new_tokens else 100.0

                print(
                    f"\rGenerating: {step:,}/{max_new_tokens:,} "
                    f"({progress:6.2f}%) "
                    f"| Elapsed {format_duration(elapsed)} "
                    f"| ETA {format_duration(eta)}",
                    end="",
                    flush=True,
                )

            if eos_id is not None and next_id == eos_id:
                break

        if show_progress:
            print()

        return generated

    def save(self, filename: str) -> None:
        """Save model configuration and all parameter values as JSON."""
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        parameters = []
        for index, parameter in enumerate(self.parameters()):
            parameters.append(
                {
                    "index": index,
                    "name": parameter.name,
                    "shape": list(parameter.shape),
                    "values": parameter.tensor.flat(),
                }
            )

        data = {
            "format": "homemade-llm-v0.2",
            "config": {
                "vocab_size": self.vocab_size,
                "d_model": self.d_model,
                "num_layers": self.num_layers,
                "hidden_dim": self.hidden_dim,
                "causal": self.causal,
            },
            "parameters": parameters,
        }

        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False)

    @classmethod
    def load(
        cls,
        filename: str,
        runtime: Optional[TensorRuntime] = None,
    ):
        """Load a model checkpoint created by save()."""
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)

        config = data["config"]

        model = cls(
            vocab_size=int(config["vocab_size"]),
            d_model=int(config["d_model"]),
            num_layers=int(config["num_layers"]),
            hidden_dim=int(config["hidden_dim"]),
            causal=bool(config.get("causal", True)),
            runtime=runtime,
            seed=0,
        )

        parameters = model.parameters()
        saved_parameters = data["parameters"]

        if len(parameters) != len(saved_parameters):
            raise ValueError(
                "Checkpoint parameter count mismatch: "
                f"model={len(parameters)}, checkpoint={len(saved_parameters)}"
            )

        for parameter, saved in zip(parameters, saved_parameters):
            saved_shape = tuple(int(x) for x in saved["shape"])

            if parameter.shape != saved_shape:
                raise ValueError(
                    "Checkpoint parameter shape mismatch: "
                    f"{parameter.shape} != {saved_shape}"
                )

            values = [
                float(value)
                for value in saved["values"]
            ]

            if len(values) != parameter.size:
                raise ValueError(
                    "Checkpoint parameter size mismatch."
                )

            parameter.runtime.memory.write(
                parameter.address,
                values,
            )

        return model

    def info(self) -> None:
        parameters = self.parameters()
        scalar_count = sum(parameter.size for parameter in parameters)

        print("Language Model")
        print("==============")
        print(f"Vocabulary size    : {self.vocab_size}")
        print(f"d_model            : {self.d_model}")
        print(f"Layers             : {self.num_layers}")
        print(f"FFN hidden dim     : {self.hidden_dim}")
        print(f"Causal             : {self.causal}")
        print(f"Parameter tensors  : {len(parameters)}")
        print(f"Parameter scalars  : {scalar_count}")


if __name__ == "__main__":
    model = LanguageModel(
        vocab_size=16,
        d_model=8,
        num_layers=2,
        hidden_dim=32,
        seed=42,
    )

    token_ids = [1, 4, 7, 2]
    logits = model(token_ids)
    print("Logits shape:", logits.shape)
    model.info()
    print("Optimizer-ready parameters:", len(model.parameters()))
