"""训练更新与只读评估分开；step=k 表示完成了 k 次更新。"""

import math

import torch
from torch.nn import functional as F

from .data import PairDataset
from .model import TrainableBigram, make_optimizer, require_finite


def metrics(nll_sum: float, pair_count: int) -> dict:
    return {"pair_count": pair_count, "nll_sum": nll_sum,
            "mean_nll": nll_sum / pair_count if pair_count else None,
            "reason": None if pair_count else "没有可评估的相邻字符对"}


@torch.no_grad()
def evaluate(model: TrainableBigram, data: PairDataset, details: bool = True) -> dict:
    logits = model(data.x)
    if data.y.dtype != torch.long or data.y.shape != data.x.shape:
        raise ValueError("y 必须是与 x 同形状的 torch.long 张量")
    if data.y.numel() and ((data.y < 0).any().item()
                          or (data.y >= model.vocab_size).any().item()):
        raise ValueError("目标 token 编号越界")
    if len(data.samples) != data.x.numel():
        raise ValueError("样本追踪与张量长度不一致")
    if data.x.numel():
        # 直接使用 logits，避免极小目标概率变成 0 后再取 log。
        losses = F.cross_entropy(logits, data.y, reduction="none")
        require_finite("评估损失", losses)
        loss_values = losses.tolist()
    else:
        loss_values = []
    result = metrics(math.fsum(loss_values), len(loss_values))
    if details:
        documents, offset = [], 0
        for document in data.documents:
            count = document["pair_count"]
            predictions = []
            for index in range(offset, offset + count):
                nll = loss_values[index]
                predictions.append({**data.samples[index],
                                    "logits": logits[index].tolist(),
                                    "probability": math.exp(-nll), "nll": nll})
            documents.append({"doc_id": document["id"],
                              "token_count": document["token_count"],
                              **metrics(math.fsum(loss_values[offset:offset + count]), count),
                              "predictions": predictions})
            offset += count
        result.update(doc_ids=[d["id"] for d in data.documents], documents=documents)
    return result


def train(model: TrainableBigram, train_data: PairDataset, val_data: PairDataset,
          learning_rate: float = 0.3, steps: int = 300, log_every: int = 10) -> dict:
    if type(steps) is not int or steps < 0:
        raise ValueError("steps 必须是非负整数，不能为布尔值")
    if type(log_every) is not int or log_every < 1:
        raise ValueError("log_every 必须是正整数，不能为布尔值")
    if not train_data.x.numel():
        raise ValueError("训练集没有相邻对，无法训练（包括 steps=0）")
    optimizer = make_optimizer(model, learning_rate)
    config = {"learning_rate": learning_rate, "steps": steps,
              "log_every": log_every, "optimizer": "SGD",
              "momentum": 0, "weight_decay": 0, "batch_mode": "full",
              "batch_size": train_data.x.numel(), "device": "cpu",
              "dtype": str(model.score_table.dtype),
              "initialization": "zeros" if (model.score_table == 0).all().item() else "provided"}
    history, first_step = [], None

    def record(step: int) -> None:
        # 两组 loss 都来自完成 step 次更新后的同一份参数。
        history.append({"step": step,
                        "train": evaluate(model, train_data, details=False),
                        "val": evaluate(model, val_data, details=False),
                        "logits": model.snapshot()})

    step = 0
    try:
        record(0)
        for step in range(1, steps + 1):
            before = model.snapshot() if step == 1 else None
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_data.x)
            loss = F.cross_entropy(logits, train_data.y)
            require_finite("训练损失", loss)
            loss.backward()
            if model.score_table.grad is None:
                raise RuntimeError("分数表没有梯度，请检查训练计算路径")
            require_finite("梯度", model.score_table.grad)
            if step == 1:
                first_step = {"before": before,
                              "after_backward": model.snapshot(),
                              "gradient": model.score_table.grad.detach().tolist(),
                              "loss_before": loss.item()}
            optimizer.step()
            require_finite("更新后的分数表", model.score_table)
            if step == 1:
                first_step["after_step"] = model.snapshot()
            if step % log_every == 0 or step == steps:
                record(step)
    except (FloatingPointError, RuntimeError, OverflowError) as error:
        raise FloatingPointError(
            f"训练失败：step={step}, lr={learning_rate}, steps={steps}, "
            f"dtype={model.score_table.dtype}；{error}"
        ) from error
    return {"config": config, "parameter_info": model.parameter_info(),
            "optimizer_manages_score_table": any(
                p is model.score_table for g in optimizer.param_groups for p in g["params"]),
            "first_step": first_step, "history": history}
