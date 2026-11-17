"""全量 batch 普通 SGD；验证按目标数加权，记录首步独立快照。"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import finite, integer, snapshot


def validate_targets(x: torch.Tensor, y: torch.Tensor, vocab_size: int) -> None:
    if (not isinstance(y, torch.Tensor) or y.dtype != torch.long
            or y.device.type != "cpu" or y.ndim != 1 or len(y) != len(x)):
        raise ValueError("标签必须为 CPU long [B]，与输入样本数相同")
    if ((y < 0) | (y >= vocab_size)).any().item():
        raise ValueError("目标 token 编号越界")


def vocab_size(model: nn.Module) -> int:
    return model.config.vocab_size if hasattr(model, "config") else model.vocab_size


def make_optimizer(model: nn.Module, learning_rate: float) -> torch.optim.SGD:
    if (type(learning_rate) not in (int, float) or not math.isfinite(learning_rate)
            or learning_rate <= 0):
        raise ValueError("learning_rate 必须是有限正数，不能为布尔值")
    parameters = list(model.parameters())
    if not parameters or any(not p.requires_grad for p in parameters):
        raise ValueError("训练模型必须具有可训练参数且本期不冻结参数")
    optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=0, weight_decay=0)
    managed = [p for group in optimizer.param_groups for p in group["params"]]
    if len(managed) != len(parameters) or {id(p) for p in managed} != {id(p) for p in parameters}:
        raise ValueError("优化器没有恰好管理模型的全部参数")
    return optimizer


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
             *, batch_size: int | None = None) -> dict:
    validate_targets(x, y, vocab_size(model))
    if batch_size is not None:
        integer("batch_size", batch_size)
    if not len(y):
        return {"count": 0, "nll_sum": 0.0, "loss": None, "accuracy": None}
    batch_size = batch_size or len(y)
    was_training = model.training
    model.eval()
    nll_sum, correct = 0.0, 0
    try:
        for start in range(0, len(y), batch_size):
            labels = y[start:start + batch_size]
            logits = model(x[start:start + batch_size])
            finite("eval logits", logits)
            loss_sum = F.cross_entropy(logits, labels, reduction="sum")
            finite("eval loss", loss_sum)
            nll_sum += float(loss_sum)
            correct += int((logits.argmax(-1) == labels).sum())
    finally:
        model.train(was_training)
    return {"count": len(y), "nll_sum": nll_sum, "loss": nll_sum / len(y),
            "accuracy": correct / len(y)}


def train(model: nn.Module, x: torch.Tensor, y: torch.Tensor, *, learning_rate: float = 0.1,
          steps: int = 1000, log_every: int = 50,
          validation: tuple[torch.Tensor, torch.Tensor] | None = None) -> dict:
    integer("steps", steps, 0)
    integer("log_every", log_every)
    validate_targets(x, y, vocab_size(model))
    if not len(y):
        raise ValueError("训练组没有可预测样本")
    optimizer = make_optimizer(model, learning_rate)
    model.train()

    def measure(step):
        return {"step": step, "train": evaluate(model, x, y),
                "val": None if validation is None else evaluate(model, *validation)}

    history = [measure(0)]
    first_step = None
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        before = snapshot(model) if step == 1 else None
        logits = model(x)
        finite(f"step {step} logits", logits)
        loss = F.cross_entropy(logits, y, reduction="mean")
        finite(f"step {step} loss", loss)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                raise RuntimeError(f"step {step}: {name} 未收到梯度，请检查训练路径")
            finite(f"step {step} {name}.grad", parameter.grad)
        if step == 1:
            first_step = {"before": before, "after_backward": snapshot(model),
                          "gradients": snapshot(model, gradients=True), "loss": float(loss.detach())}
        optimizer.step()
        for name, parameter in model.named_parameters():
            finite(f"step {step} updated {name}", parameter)
        if step == 1:
            first_step["after_step"] = snapshot(model)
        if step % log_every == 0 or step == steps:
            history.append(measure(step))
    return {"learning_rate": learning_rate, "steps": steps, "log_every": log_every,
            "optimizer": {"name": "SGD", "momentum": 0, "weight_decay": 0},
            "history": history, "first_step": first_step}
