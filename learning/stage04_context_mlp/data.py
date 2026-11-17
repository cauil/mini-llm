"""只读复用阶段 0C 窗口，适配为每个上下文一个目标。"""

from dataclasses import dataclass

import torch

from learning.stage00_char_codec.tokenizer import CharTokenizer
from learning.stage00_dataset.prepare import prepare_dataset


@dataclass
class ContextData:
    x: torch.Tensor
    y: torch.Tensor
    samples: list[dict]

    def to_dict(self) -> dict:
        return {"x": self.x.tolist(), "y": self.y.tolist(),
                "count": len(self.samples), "samples": self.samples}


def prepare_data(raw: object, tokenizer: CharTokenizer, k: int) -> tuple[dict, dict]:
    windows = prepare_dataset(raw, tokenizer, k)
    groups = {}
    for split in ("train", "val"):
        samples = [{"split": split, "doc_id": s["doc_id"], "start": s["start"],
                    "context_ids": list(s["x"]), "target_id": s["y"][-1],
                    "target_position": s["start"] + k,
                    "context_text": tokenizer.decode(s["x"]),
                    "target_text": tokenizer.decode([s["y"][-1]])}
                   for s in windows[split]]
        x = torch.tensor([s["context_ids"] for s in samples], dtype=torch.long).reshape(-1, k)
        y = torch.tensor([s["target_id"] for s in samples], dtype=torch.long)
        groups[split] = ContextData(x, y, samples)
    return groups, windows["metadata"]
