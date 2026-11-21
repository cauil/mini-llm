"""仅用于证明检查有效的故障变体；正式模型不导入本文件。"""

import math
import torch


def broken_attention(model, x, *, fault: str):
    q, k, v = model.project(x)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    mask = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device)
    if fault == "no_mask":
        masked = scores
    elif fault == "reversed_mask":
        masked = scores.masked_fill(mask.tril(-1), -torch.inf)
    elif fault == "query_softmax":
        masked = scores.masked_fill(mask.triu(1), -torch.inf)
    else:
        raise ValueError(f"未知故障：{fault}")
    weights = masked.softmax(dim=-2 if fault == "query_softmax" else -1)
    return weights @ v, weights
