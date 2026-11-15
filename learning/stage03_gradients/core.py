"""数值参考、中心差分和检查记录；不依赖 autograd 构造预期梯度。"""

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F

DTYPE = torch.float64
AUTOGRAD_ATOL = 1e-10
DIFFERENCE_ATOL = 1e-7
DEFAULT_H = 1e-5


def finite_number(value, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} 必须是有限实数，不能为布尔值")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} 超出浮点表示范围") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} 必须有限")
    return result


def positive_number(value, name: str) -> float:
    result = finite_number(value, name)
    if result <= 0:
        raise ValueError(f"{name} 必须为正")
    return result


def validate_sample(logits, target) -> list[float]:
    if not isinstance(logits, (list, tuple)) or not logits:
        raise ValueError("logits 必须是非空的一维列表或元组")
    scores = [finite_number(v, "logits 分量") for v in logits]
    if type(target) is not int or not 0 <= target < len(scores):
        raise ValueError("target 必须是范围内的整数编号，不能为布尔值")
    return scores


def ensure_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"{name} 出现 NaN/Inf")


def values(tensor: torch.Tensor):
    ensure_finite(tensor, "保存的张量")
    # tolist 返回独立的 Python 数值；不与后续原地更新共享存储。
    return tensor.detach().cpu().tolist()


def snapshot(parameter: torch.Tensor) -> dict:
    return {
        "parameter": values(parameter),
        "grad": None if parameter.grad is None else values(parameter.grad),
    }


def reference(logits, target) -> dict:
    """用 Python math 独立计算稳定 CE、概率及 p-one_hot。"""
    scores = validate_sample(logits, target)
    shifted = [v - max(scores) for v in scores]
    weights = [math.exp(v) for v in shifted]
    denominator = math.fsum(weights)
    probabilities = [v / denominator for v in weights]
    loss = math.log(denominator) - shifted[target]
    gradient = [p - int(i == target) for i, p in enumerate(probabilities)]
    if not math.isfinite(loss):
        raise FloatingPointError("参考损失超出浮点表示范围")
    return {"probabilities": probabilities, "loss": loss, "gradient": gradient}


def cross_entropy(z: torch.Tensor, target: int) -> torch.Tensor:
    label = torch.tensor([target], dtype=torch.long, device="cpu")
    loss = F.cross_entropy(z.unsqueeze(0), label)
    ensure_finite(loss, "交叉熵")
    return loss


def make_sgd(parameters, learning_rate):
    lr = positive_number(learning_rate, "learning_rate")
    return torch.optim.SGD(parameters, lr=lr, momentum=0, weight_decay=0)


def central_difference(
    base: torch.Tensor, loss_fn: Callable[[torch.Tensor], torch.Tensor], h=DEFAULT_H
) -> torch.Tensor:
    """对任意形状浮点参数逐格扰动，返回相同形状的数值梯度。"""
    h = positive_number(h, "h")
    if base.device.type != "cpu" or base.dtype != DTYPE or base.numel() == 0:
        raise ValueError("差分参数必须是非空 CPU float64 张量")
    ensure_finite(base, "差分基点")
    point = base.detach().clone().contiguous()
    result = torch.empty_like(point)
    with torch.no_grad():
        for i in range(point.numel()):
            plus, minus = point.clone(), point.clone()
            plus.view(-1)[i] += h
            minus.view(-1)[i] -= h
            ensure_finite(plus, "正向扰动")
            ensure_finite(minus, "负向扰动")
            lp, lm = loss_fn(plus), loss_fn(minus)
            if lp.numel() != 1 or lm.numel() != 1:
                raise ValueError("差分目标必须是标量损失")
            ensure_finite(lp, "正侧损失")
            ensure_finite(lm, "负侧损失")
            result.view(-1)[i] = (lp.item() - lm.item()) / (2 * h)
    ensure_finite(result, "差分梯度")
    return result


@dataclass
class Checks:
    entries: list[dict] = field(default_factory=list)

    def truth(self, name: str, condition: bool, detail: str = "") -> None:
        self.entries.append({"name": name, "passed": bool(condition), "detail": detail})

    def close(self, name: str, actual, expected, atol=AUTOGRAD_ATOL) -> None:
        a = torch.as_tensor(actual, dtype=DTYPE)
        e = torch.as_tensor(expected, dtype=DTYPE)
        same_shape = a.shape == e.shape
        finite = bool(torch.isfinite(a).all() and torch.isfinite(e).all())
        error = (a - e).abs().max().item() if same_shape and finite and a.numel() else None
        self.entries.append({
            "name": name, "passed": error is not None and error <= atol,
            "max_absolute_error": error, "absolute_tolerance": atol,
            "detail": "" if same_shape and finite else "形状不符或出现非有限数值",
        })

    def summary(self) -> dict:
        failed = [item["name"] for item in self.entries if not item["passed"]]
        return {"total": len(self.entries), "passed": len(self.entries) - len(failed),
                "failed": len(failed), "failed_names": failed}


def compare_sample(logits, target, h=DEFAULT_H) -> dict:
    """公开的单样本对照入口，接受新的词表大小与目标。"""
    scores = validate_sample(logits, target)
    h = positive_number(h, "h")
    expected = reference(scores, target)
    z = torch.tensor(scores, dtype=DTYPE, requires_grad=True)
    before = snapshot(z)
    numerical = central_difference(z, lambda v: cross_entropy(v, target), h)
    after_difference = snapshot(z)
    loss = cross_entropy(z, target)
    loss.backward()
    checks = Checks()
    checks.close("解析与autograd", values(z.grad), expected["gradient"])
    checks.close("解析与中心差分", values(numerical), expected["gradient"], DIFFERENCE_ATOL)
    checks.close("独立参考损失", loss.item(), expected["loss"])
    checks.truth("差分保持基点及梯度", before == after_difference)
    checks.truth("backward不改参数", values(z) == before["parameter"])
    return {
        "logits": scores, "target": target, "h": h,
        "probabilities": expected["probabilities"], "loss": loss.item(),
        "analytic_gradient": expected["gradient"], "difference_gradient": values(numerical),
        "autograd_gradient": values(z.grad), "before": before,
        "after_difference": after_difference, "after_backward": snapshot(z),
        "checks": checks.entries, "summary": checks.summary(),
    }
