"""字符级计数 bigram：逐文档计数、加一平滑、生成与平均 NLL。"""

import math
import random
from typing import TypedDict

from learning.stage00_char_codec.tokenizer import CharTokenizer


class Document(TypedDict):
    id: str
    text: str


def validate_documents(documents: object, seen=None) -> list[Document]:
    """复制并校验文档；共享 seen 时也能检查 train/val 跨组重复 id。"""
    if not isinstance(documents, list):
        raise ValueError("文档必须是列表")
    if seen is None:
        seen = set()
    checked = []
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError(f"文档 [{index}] 必须是包含 id 和 text 的对象")
        doc_id, text = document.get("id"), document.get("text")
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError(f"文档 [{index}] 的 id 必须为非空字符串")
        if doc_id in seen:
            raise ValueError(f"文档 id 重复（含跨组重复）：{doc_id!r}")
        if not isinstance(text, str):
            raise ValueError(f"文档 {doc_id!r} 的 text 必须为字符串")
        seen.add(doc_id)
        checked.append({"id": doc_id, "text": text})
    return checked


def validate_groups(data: object) -> dict[str, list[Document]]:
    if not isinstance(data, dict) or set(data) != {"train", "val"}:
        raise ValueError("数据必须是只含 train 和 val 两组的对象")
    seen = set()
    return {split: validate_documents(data[split], seen) for split in ("train", "val")}


def _metrics(loss_sum: float, pair_count: int) -> dict:
    return {
        "pair_count": pair_count,
        "nll_sum": loss_sum,
        "mean_nll": loss_sum / pair_count if pair_count else None,
        "reason": None if pair_count else "没有可评估的相邻字符对",
    }


class CountBigram:
    """fit 只接收训练文档；evaluate 与 generate 均不更新模型。"""

    def __init__(self, tokenizer: CharTokenizer) -> None:
        self._tokenizer = tokenizer
        self._size = len(tokenizer.vocab)
        self._counts = None
        self._probabilities = None

    def _require_fitted(self) -> None:
        if self._counts is None:
            raise ValueError("请先调用 fit；空训练列表也可以显式拟合")

    def _encode_documents(self, documents: list[Document]) -> list:
        encoded = []
        for document in validate_documents(documents):
            try:
                tokens = self._tokenizer.encode(document["text"])
            except KeyError as error:
                raise ValueError(
                    f"文档 {document['id']!r} 含未知字符：{error.args[0]!r}"
                ) from error
            encoded.append((document["id"], tokens))
        return encoded

    def fit(self, documents: list[Document]) -> "CountBigram":
        # 先完整编码；失败时保留上一次成功拟合的状态。
        encoded = self._encode_documents(documents)
        counts = [[0] * self._size for _ in range(self._size)]
        for _, tokens in encoded:
            # 每篇独立枚举相邻位置，不跨文档，也不重复累计重叠窗口。
            for current, target in zip(tokens, tokens[1:]):
                counts[current][target] += 1
        probabilities = []
        for row in counts:
            total = sum(row) + self._size
            probabilities.append([(count + 1) / total for count in row])
        self._counts, self._probabilities = counts, probabilities
        return self

    @property
    def counts(self) -> list[list[int]]:
        self._require_fitted()
        return [row[:] for row in self._counts]

    @property
    def probabilities(self) -> list[list[float]]:
        self._require_fitted()
        return [row[:] for row in self._probabilities]

    def distribution(self, current_id: int) -> list[float]:
        self._require_fitted()
        if type(current_id) is not int or not 0 <= current_id < self._size:
            raise ValueError("当前 token 编号必须是词表内的整数，不能为布尔值")
        return self._probabilities[current_id][:]

    def sample_next(self, current_id: int, rng: random.Random) -> int:
        """调用者提供持续使用的随机数生成器，也可固定一行做频率检查。"""
        return rng.choices(range(self._size), weights=self.distribution(current_id), k=1)[0]

    def generate(self, prompt: str, num_new_tokens: int,
                 method: str = "sample", seed: int = 0) -> dict:
        self._require_fitted()
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("提示词必须是非空字符串")
        if type(num_new_tokens) is not int or num_new_tokens < 0:
            raise ValueError("新增字符数必须是非负整数，不能为布尔值")
        if method not in ("argmax", "sample"):
            raise ValueError("生成方法必须为 argmax 或 sample")
        if type(seed) is not int:
            raise ValueError("种子必须是整数，不能为布尔值")
        try:
            tokens = self._tokenizer.encode(prompt)
        except KeyError as error:
            raise ValueError(f"提示词含未知字符：{error.args[0]!r}") from error
        rng = random.Random(seed)
        for _ in range(num_new_tokens):
            current = tokens[-1]
            if method == "argmax":
                row = self.distribution(current)
                # range 按编号递增，max 在并列时返回第一个编号。
                next_id = max(range(self._size), key=row.__getitem__)
            else:
                next_id = self.sample_next(current, rng)
            tokens.append(next_id)
        return {
            "prompt": prompt, "num_new_tokens": num_new_tokens,
            "method": method, "seed": seed,
            "text": self._tokenizer.decode(tokens), "tokens": tokens,
        }

    def evaluate(self, documents: list[Document]) -> dict:
        self._require_fitted()
        results = []
        for doc_id, tokens in self._encode_documents(documents):
            predictions = []
            for position in range(1, len(tokens)):
                current, target = tokens[position - 1], tokens[position]
                probability = self._probabilities[current][target]
                predictions.append({
                    "target_position": position,
                    "current_id": current, "target_id": target,
                    "probability": probability, "nll": -math.log(probability),
                })
            loss_sum = math.fsum(item["nll"] for item in predictions)
            results.append({
                "doc_id": doc_id, "token_count": len(tokens),
                **_metrics(loss_sum, len(predictions)), "predictions": predictions,
            })
        # 按全部预测位置平均，不能直接平均长短不同文档的 mean_nll。
        return {
            **_metrics(math.fsum(doc["nll_sum"] for doc in results),
                       sum(doc["pair_count"] for doc in results)),
            "doc_ids": [doc["doc_id"] for doc in results],
            "documents": results,
        }

    def to_dict(self) -> dict:
        return {
            "vocab": self._tokenizer.vocab,
            "counts": self.counts,
            "probabilities": self.probabilities,
            "smoothing": {"name": "add_one", "alpha": 1, "normalization": "row"},
        }
