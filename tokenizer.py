# tokenizer.py
#
# Simple character-level tokenizer
#
# Purpose:
#   - Build vocabulary from text
#   - Convert text -> token IDs
#   - Convert token IDs -> text
#   - Support special tokens
#   - Save / load vocabulary
#
# No external libraries are required.

import json
from typing import List, Dict, Optional


class Tokenizer:
    """
    Simple character-level tokenizer.

    Each character becomes one token.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.fitted = False
        self._initialize_special_tokens()

    def _initialize_special_tokens(self) -> None:
        special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
        ]
        for token in special_tokens:
            self._add_token(token)

    def _add_token(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]

        token_id = len(self.token_to_id)
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        return token_id

    def fit(self, text: str) -> None:
        unique_chars = sorted(set(text))
        for char in unique_chars:
            self._add_token(char)
        self.fitted = True

    def fit_texts(self, texts: List[str]) -> None:
        all_text = ""
        for text in texts:
            all_text += text
        self.fit(all_text)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        if not self.fitted:
            raise RuntimeError(
                "Tokenizer vocabulary has not been fitted yet."
            )

        result: List[int] = []

        if add_bos:
            result.append(self.bos_id)

        for char in text:
            token_id = self.token_to_id.get(char, self.unk_id)
            result.append(token_id)

        if add_eos:
            result.append(self.eos_id)

        return result

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        result = []
        special_tokens = {
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
        }

        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id), self.UNK_TOKEN)

            if skip_special_tokens and token in special_tokens:
                continue

            result.append(token)

        return "".join(result)

    def token_to_index(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_id)

    def index_to_token(self, token_id: int) -> str:
        return self.id_to_token.get(token_id, self.UNK_TOKEN)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS_TOKEN]

    def pad(self, token_ids: List[int], max_length: int) -> List[int]:
        if max_length <= 0:
            raise ValueError("max_length must be > 0.")

        if len(token_ids) >= max_length:
            return token_ids[:max_length]

        padding_length = max_length - len(token_ids)
        return token_ids + [self.pad_id] * padding_length

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[List[int]]:
        result = []

        for text in texts:
            ids = self.encode(
                text,
                add_bos=add_bos,
                add_eos=add_eos,
            )

            if max_length is not None:
                ids = self.pad(ids, max_length)

            result.append(ids)

        return result

    def save(self, filename: str) -> None:
        data = {
            "token_to_id": self.token_to_id,
            "fitted": self.fitted,
        }

        with open(filename, "w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, filename: str):
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)

        tokenizer = cls()

        tokenizer.token_to_id = {
            token: int(token_id)
            for token, token_id in data["token_to_id"].items()
        }

        tokenizer.id_to_token = {
            token_id: token
            for token, token_id in tokenizer.token_to_id.items()
        }

        tokenizer.fitted = data.get("fitted", True)

        return tokenizer

    def dump_vocab(self) -> None:
        print("Tokenizer Vocabulary")
        print("====================")

        for token_id in sorted(self.id_to_token.keys()):
            token = self.id_to_token[token_id]
            print(f"{token_id:4d}: {repr(token)}")


if __name__ == "__main__":
    print()
    print("================================")
    print("Tokenizer Test")
    print("================================")
    print()

    text = "hello world"

    tokenizer = Tokenizer()
    tokenizer.fit(text)

    print("Vocabulary size:", tokenizer.vocab_size)
    print()

    tokenizer.dump_vocab()
    print()

    sample = "hello"

    ids = tokenizer.encode(
        sample,
        add_bos=True,
        add_eos=True,
    )

    print("Text:", sample)
    print("Token IDs:", ids)

    decoded = tokenizer.decode(ids)
    print("Decoded:", decoded)
    print()

    padded = tokenizer.pad(ids, max_length=12)
    print("Padded:", padded)
    print()

    batch = tokenizer.batch_encode(
        ["hello", "world"],
        max_length=10,
        add_bos=True,
        add_eos=True,
    )

    print("Batch:")
    for row in batch:
        print(row)

    print()

    tokenizer.save("tokenizer.json")
    print("Saved tokenizer.json")

    tokenizer2 = Tokenizer.load("tokenizer.json")
    test_ids = tokenizer2.encode("world")

    print("Reloaded encode:", test_ids)
    print("Reloaded decode:", tokenizer2.decode(test_ids))
