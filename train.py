# train.py
#
# Training loop for the homemade LLM.
#
# Dependencies:
#   tokenizer.py
#   llm.py
#   loss.py
#   optimizer.py
#
# Current status:
#   - Forward pass works.
#   - Cross entropy loss works.
#   - Parameters can be collected.
#   - Optimizer exists.
#   - Full end-to-end autograd through LanguageModel is not connected yet.
#
# Therefore optimizer.step() is intentionally disabled until the model
# forward path is converted to AutoTensor-compatible operations.
#
# v0.1 memory-management note:
#   The virtual GPU uses a linear allocator. Temporary tensors created by a
#   forward pass would otherwise consume memory permanently. Trainer records
#   the end of the persistent model allocation and rewinds temporary virtual
#   GPU allocations after each step.

from typing import List, Tuple

from tokenizer import Tokenizer
from llm import LanguageModel
from loss import CrossEntropyLoss, shift_tokens, perplexity
from optimizer import Adam


def build_training_samples(
    token_ids: List[int],
    context_length: int,
) -> List[Tuple[List[int], List[int]]]:
    """Create next-token language-model training samples."""

    if context_length <= 0:
        raise ValueError("context_length must be greater than 0.")

    if len(token_ids) < 2:
        raise ValueError("At least two tokens are required.")

    samples = []
    maximum_start = len(token_ids) - context_length - 1

    if maximum_start < 0:
        inputs, targets = shift_tokens(token_ids)
        samples.append((inputs, targets))
        return samples

    for start in range(maximum_start + 1):
        window = token_ids[start:start + context_length + 1]
        inputs = window[:-1]
        targets = window[1:]
        samples.append((inputs, targets))

    return samples


class Trainer:
    """Simple language-model trainer scaffold."""

    def __init__(
        self,
        model: LanguageModel,
        learning_rate: float = 1e-3,
    ):
        self.model = model
        self.loss_fn = CrossEntropyLoss(reduction="mean")
        self.optimizer = Adam(
            self.model.parameters(),
            learning_rate=learning_rate,
        )

        # Model parameters have already been allocated at this point.
        # Everything allocated after this address during forward/loss is
        # temporary in the current v0.1 execution model.
        self._temporary_memory_mark = (
            self.model.runtime.memory.next_address
        )

    def _release_temporary_gpu_memory(self) -> None:
        """Rewind virtual GPU allocations created after model construction.

        The v0.1 GPUMemory allocator is intentionally simple and does not yet
        reuse freed blocks.  A training forward pass creates many temporary
        Tensor objects, so without this rewind every sample would permanently
        advance next_address and eventually exhaust the simulated GPU memory.

        Persistent model parameters are all below _temporary_memory_mark and
        are therefore preserved.
        """

        memory = self.model.runtime.memory
        mark = self._temporary_memory_mark

        temporary_addresses = [
            address
            for address in memory.allocations
            if address >= mark
        ]

        for address in temporary_addresses:
            del memory.allocations[address]

        memory.next_address = mark

    def forward_step(
        self,
        input_ids: List[int],
        target_ids: List[int],
    ) -> float:
        try:
            logits = self.model(input_ids)
            return self.loss_fn(logits, target_ids)
        finally:
            self._release_temporary_gpu_memory()

    def train_step(
        self,
        input_ids: List[int],
        target_ids: List[int],
    ) -> float:
        """
        Execute one training-step scaffold.

        Backward and optimizer.step() are intentionally disabled until
        LanguageModel forward operations participate in the autograd graph.
        """

        self.optimizer.zero_grad()

        try:
            logits = self.model(input_ids)
            loss = self.loss_fn(logits, target_ids)

            # Future end-to-end autograd path:
            # loss.backward()
            # self.optimizer.step()

            return loss
        finally:
            # The current loss is a Python float, so all tensors allocated by
            # this forward/loss calculation are temporary and can be discarded.
            self._release_temporary_gpu_memory()

    def train(
        self,
        samples: List[Tuple[List[int], List[int]]],
        epochs: int = 10,
        verbose: bool = True,
    ) -> List[float]:
        if epochs <= 0:
            raise ValueError("epochs must be greater than 0.")

        if len(samples) == 0:
            raise ValueError("samples must not be empty.")

        history = []

        for epoch in range(epochs):
            total_loss = 0.0

            for input_ids, target_ids in samples:
                total_loss += self.train_step(input_ids, target_ids)

            average_loss = total_loss / len(samples)
            history.append(average_loss)

            if verbose:
                print(
                    f"Epoch {epoch + 1:4d} "
                    f"| Loss {average_loss:.6f} "
                    f"| Perplexity {perplexity(average_loss):.6f}"
                )

        return history


def train_text(
    text: str,
    d_model: int = 16,
    num_layers: int = 2,
    hidden_dim: int = 64,
    context_length: int = 8,
    epochs: int = 10,
    learning_rate: float = 1e-3,
):
    """Build tokenizer, model, dataset and trainer from plain text."""

    if not text:
        raise ValueError("Training text must not be empty.")

    tokenizer = Tokenizer()
    tokenizer.fit(text)

    token_ids = tokenizer.encode(
        text,
        add_bos=True,
        add_eos=True,
    )

    samples = build_training_samples(
        token_ids,
        context_length=context_length,
    )

    model = LanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        causal=True,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        learning_rate=learning_rate,
    )

    history = trainer.train(
        samples,
        epochs=epochs,
    )

    return tokenizer, model, history


if __name__ == "__main__":
    print()
    print("================================")
    print("Homemade LLM Training Test")
    print("================================")
    print()

    text = (
        "hello world "
        "hello llm "
        "hello gpu "
        "world hello "
    )

    tokenizer = Tokenizer()
    tokenizer.fit(text)

    token_ids = tokenizer.encode(
        text,
        add_bos=True,
        add_eos=True,
    )

    context_length = 8
    samples = build_training_samples(
        token_ids,
        context_length,
    )

    model = LanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=8,
        num_layers=1,
        hidden_dim=32,
        seed=42,
    )

    print("Vocabulary size:", tokenizer.vocab_size)
    print("Token count:", len(token_ids))
    print("Number of samples:", len(samples))
    print("Trainable parameter tensors:", len(model.parameters()))
    print()

    trainer = Trainer(
        model=model,
        learning_rate=1e-3,
    )

    history = trainer.train(
        samples,
        epochs=3,
    )

    print()
    print("Loss history:", history)
    print()
    print(
        "NOTE: model forward operations are not yet connected to AutoTensor; "
        "optimizer.step() is intentionally disabled."
    )
