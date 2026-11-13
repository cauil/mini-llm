"""字符级编码器：固定词表、查表转换和 JSON 保存加载。"""

import json
from pathlib import Path


def _validate_vocab(vocab: dict[str, int]) -> None:
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError(f"词表必须是非空字典，收到：{vocab!r}")
    for char, token_id in vocab.items():
        if not isinstance(char, str) or len(char) != 1:
            raise ValueError(f"词表 key 必须是一个字符，收到：{char!r}")
        # bool 在 Python 中是 int 的子类，但这里不能把 True 当作编号 1。
        if type(token_id) is not int:
            raise ValueError(f"字符 {char!r} 的编号必须是整数，收到：{token_id!r}")
    ids = list(vocab.values())
    if len(set(ids)) != len(ids):
        raise ValueError(f"词表包含重复编号：{ids!r}")
    if set(ids) != set(range(len(vocab))):
        raise ValueError(f"词表编号必须连续覆盖 0 到 {len(vocab) - 1}，收到：{ids!r}")


class CharTokenizer:
    """创建时保存词表副本；后续调用复用同一份字符与编号映射。"""

    def __init__(self, vocab: dict[str, int]) -> None:
        _validate_vocab(vocab)
        self._vocab = dict(vocab)
        self._inv_vocab = {token_id: char for char, token_id in self._vocab.items()}

    @property
    def vocab(self) -> dict[str, int]:
        """返回副本，外部修改它不会改变编码器。"""
        return dict(self._vocab)

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text 必须是字符串")
        # 未知字符保留字典的 KeyError，异常中包含该字符。
        return [self._vocab[char] for char in text]

    def decode(self, ids: list[int]) -> str:
        for token_id in ids:
            if type(token_id) is not int:
                raise ValueError(f"编号必须是整数，收到：{token_id!r}")
        # 负编号、越界编号同样通过 KeyError 明确拒绝。
        return "".join(self._inv_vocab[token_id] for token_id in ids)

    def save(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self._vocab, file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as file:
            vocab = json.load(file)
        # 加载后走同一入口，避免绕过词表校验。
        return cls(vocab)


# 保留旧函数接口：每次按传入词表独立工作，不跨调用保存状态。
# 需要固定映射时，应创建一个 CharTokenizer 并复用它。
def encode(text: str, vocab: dict[str, int]) -> list[int]:
    return CharTokenizer(vocab).encode(text)


def decode(ids: list[int], vocab: dict[str, int]) -> str:
    return CharTokenizer(vocab).decode(ids)


def save_vocab(vocab: dict[str, int], path: Path) -> None:
    CharTokenizer(vocab).save(path)


def load_vocab(path: Path) -> dict[str, int]:
    return CharTokenizer.load(path).vocab
