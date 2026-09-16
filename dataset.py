# dataset.py
#
# Simple dataset utilities for a small language model.
#
# Responsibilities:
#   - Load UTF-8 text files
#   - Convert text into token IDs using Tokenizer
#   - Create next-token prediction samples
#   - Create mini-batches
#   - Split data into train / validation / test sets

from __future__ import annotations

import os
import random
from typing import Iterable, List, Sequence, Tuple

import numpy as np


class TextDataset:
    """
    Dataset for autoregressive next-token prediction.

    Given token IDs:

        [BOS, A, B, C, D, EOS]

    and sequence_length = 3,

    samples become:

        input : [BOS, A, B]
        target: [A, B, C]

        input : [A, B, C]
        target: [B, C, D]

        input : [B, C, D]
        target: [C, D, EOS]
    """

    def __init__(
        self,
        text: str,
        tokenizer,
        sequence_length: int = 64,
        stride: int = 1,
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        if sequence_length <= 0:
            raise ValueError("sequence_length must be greater than 0")

        if stride <= 0:
            raise ValueError("stride must be greater than 0")

        self.text = text
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.stride = stride

        self.token_ids = self._encode_text(
            text,
            add_bos=add_bos,
            add_eos=add_eos,
        )

        minimum_tokens = sequence_length + 1

        if len(self.token_ids) < minimum_tokens:
            raise ValueError(
                f"Not enough tokens. Need at least {minimum_tokens}, "
                f"but got {len(self.token_ids)}."
            )

        self.sample_positions = list(
            range(
                0,
                len(self.token_ids) - sequence_length,
                stride,
            )
        )

    @staticmethod
    def load_text(filename: str, encoding: str = "utf-8") -> str:
        """Load a UTF-8 text file."""
        with open(filename, "r", encoding=encoding) as f:
            return f.read()

    @staticmethod
    def load_texts(
        filenames: Sequence[str],
        encoding: str = "utf-8",
        separator: str = "\n",
    ) -> str:
        """Load multiple text files and concatenate them."""
        texts = []

        for filename in filenames:
            with open(filename, "r", encoding=encoding) as f:
                texts.append(f.read())

        return separator.join(texts)

    @staticmethod
    def load_directory(
        directory: str,
        extension: str = ".txt",
        encoding: str = "utf-8",
        separator: str = "\n",
        recursive: bool = True,
    ) -> str:
        """Load every text file in a directory."""
        filenames = []

        if recursive:
            for root, _, files in os.walk(directory):
                for filename in files:
                    if filename.endswith(extension):
                        filenames.append(os.path.join(root, filename))
        else:
            for filename in os.listdir(directory):
                path = os.path.join(directory, filename)
                if os.path.isfile(path) and filename.endswith(extension):
                    filenames.append(path)

        filenames.sort()

        return TextDataset.load_texts(
            filenames,
            encoding=encoding,
            separator=separator,
        )

    def _encode_text(
        self,
        text: str,
        add_bos: bool,
        add_eos: bool,
    ) -> List[int]:
        """Encode text using the project's Tokenizer."""
        token_ids = list(self.tokenizer.encode(text))

        if add_bos:
            bos_id = self._get_special_token_id(
                "bos_id",
                "BOS_ID",
                "bos_token_id",
            )
            if bos_id is not None:
                token_ids.insert(0, bos_id)

        if add_eos:
            eos_id = self._get_special_token_id(
                "eos_id",
                "EOS_ID",
                "eos_token_id",
            )
            if eos_id is not None:
                token_ids.append(eos_id)

        return token_ids

    def _get_special_token_id(self, *names):
        """Find a special token ID with tolerant naming conventions."""
        for name in names:
            if hasattr(self.tokenizer, name):
                value = getattr(self.tokenizer, name)
                if callable(value):
                    value = value()
                return value

        fallback = {
            "bos_id": 2,
            "BOS_ID": 2,
            "bos_token_id": 2,
            "eos_id": 3,
            "EOS_ID": 3,
            "eos_token_id": 3,
        }

        for name in names:
            if name in fallback:
                return fallback[name]

        return None

    def __len__(self) -> int:
        """Return number of training samples."""
        return len(self.sample_positions)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return one training sample."""
        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError("dataset index out of range")

        start = self.sample_positions[index]
        end = start + self.sequence_length

        input_ids = self.token_ids[start:end]
        target_ids = self.token_ids[start + 1 : end + 1]

        return (
            np.asarray(input_ids, dtype=np.int64),
            np.asarray(target_ids, dtype=np.int64),
        )

    def batch_iter(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """Iterate over mini-batches."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        indices = list(range(len(self)))

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]

            if drop_last and len(batch_indices) < batch_size:
                break

            inputs = []
            targets = []

            for index in batch_indices:
                x, y = self[index]
                inputs.append(x)
                targets.append(y)

            input_batch = np.stack(inputs, axis=0)
            target_batch = np.stack(targets, axis=0)

            yield input_batch, target_batch

    @property
    def num_tokens(self) -> int:
        """Number of encoded tokens."""
        return len(self.token_ids)

    @property
    def num_samples(self) -> int:
        """Number of generated sequences."""
        return len(self)

    def info(self) -> dict:
        """Return basic dataset statistics."""
        return {
            "characters": len(self.text),
            "tokens": len(self.token_ids),
            "samples": len(self),
            "sequence_length": self.sequence_length,
            "stride": self.stride,
        }


def split_text(
    text: str,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[str, str, str]:
    """Split a text corpus into train / validation / test sets."""
    total = train_ratio + validation_ratio + test_ratio

    if not np.isclose(total, 1.0):
        raise ValueError(
            "train_ratio + validation_ratio + test_ratio must equal 1.0"
        )

    if train_ratio < 0 or validation_ratio < 0 or test_ratio < 0:
        raise ValueError("split ratios must be non-negative")

    length = len(text)
    train_end = int(length * train_ratio)
    validation_end = train_end + int(length * validation_ratio)

    train_text = text[:train_end]
    validation_text = text[train_end:validation_end]
    test_text = text[validation_end:]

    return train_text, validation_text, test_text


def create_datasets(
    text: str,
    tokenizer,
    sequence_length: int = 64,
    stride: int = 1,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    """Create train / validation / test TextDataset objects."""
    train_text, validation_text, test_text = split_text(
        text,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )

    train_dataset = TextDataset(
        train_text,
        tokenizer,
        sequence_length=sequence_length,
        stride=stride,
    )

    validation_dataset = TextDataset(
        validation_text,
        tokenizer,
        sequence_length=sequence_length,
        stride=stride,
    )

    test_dataset = TextDataset(
        test_text,
        tokenizer,
        sequence_length=sequence_length,
        stride=stride,
    )

    return train_dataset, validation_dataset, test_dataset


if __name__ == "__main__":
    try:
        from tokenizer import Tokenizer
    except ImportError:
        print("tokenizer.py was not found.")
        raise

    sample_text = (
        "Hello, world.\n"
        "This is a small language model.\n"
        "The model learns to predict the next token.\n"
    )

    tokenizer = Tokenizer()
    tokenizer.fit(sample_text)

    dataset = TextDataset(
        text=sample_text,
        tokenizer=tokenizer,
        sequence_length=8,
        stride=1,
    )

    print("Dataset information:")
    print(dataset.info())
    print()

    x, y = dataset[0]

    print("First sample:")
    print("input :", x)
    print("target:", y)
    print()

    print("Decoded input:")
    print(tokenizer.decode(x.tolist()))
    print()

    print("Mini-batches:")

    for x_batch, y_batch in dataset.batch_iter(
        batch_size=4,
        shuffle=False,
    ):
        print("input shape :", x_batch.shape)
        print("target shape:", y_batch.shape)
        break
