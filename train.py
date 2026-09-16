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

    def forward_step(
        self,
        input_ids: List[int],
        target_ids: List[int],
    ) -> float:
        logits = self.model(input_ids)
        return self.loss_fn(logits, target_ids)

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

        logits = self.model(input_ids)
        loss = self.loss_fn(logits, target_ids)

        # Future end-to-end autograd path:
        # loss.backward()
        # self.optimizer.step()

        return loss

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
