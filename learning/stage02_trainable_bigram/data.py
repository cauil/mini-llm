"""复用词表和文档校验，逐篇枚举一次相邻对。"""

from dataclasses import dataclass

import torch

from learning.stage00_char_codec.tokenizer import CharTokenizer
from learning.stage01_count_bigram.bigram import validate_groups


@dataclass
class PairDataset:
    documents: list[dict]
    samples: list[dict]
    x: torch.Tensor
    y: torch.Tensor

    def to_dict(self) -> dict:
        return {"documents": self.documents, "samples": self.samples,
                "pair_count": len(self.samples),
                "x": self.x.tolist(), "y": self.y.tolist()}


def prepare_data(tokenizer: CharTokenizer, raw: object) -> dict[str, PairDataset]:
    groups = validate_groups(raw)
    prepared = {}
    for split, documents in groups.items():
        encoded, samples = [], []
        for document in documents:
            try:
                tokens = tokenizer.encode(document["text"])
            except KeyError as error:
                raise ValueError(
                    f"文档 {document['id']!r} 含未知字符：{error.args[0]!r}"
                ) from error
            encoded.append({**document, "tokens": tokens,
                            "token_count": len(tokens),
                            "pair_count": max(0, len(tokens) - 1)})
            # 每篇独立，不把上一篇末尾与下一篇开头连接。
            for position in range(len(tokens) - 1):
                samples.append({"doc_id": document["id"],
                                "input_position": position,
                                "target_position": position + 1,
                                "current_id": tokens[position],
                                "target_id": tokens[position + 1]})
        prepared[split] = PairDataset(
            encoded, samples,
            torch.tensor([s["current_id"] for s in samples], dtype=torch.long),
            torch.tensor([s["target_id"] for s in samples], dtype=torch.long),
        )
    return prepared
