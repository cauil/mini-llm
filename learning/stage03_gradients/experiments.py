"""任务书 A–F：输入集中在此，每组实验独立初始化。"""

import math

import torch
import torch.nn.functional as F

from learning.stage02_trainable_bigram.model import TrainableBigram
from .core import (
    Checks, DEFAULT_H, DIFFERENCE_ATOL, DTYPE, central_difference,
    compare_sample, cross_entropy, ensure_finite, make_sgd, reference, snapshot, values,
)

SAMPLES = [
    ("A1", [0, 0, 0], 1),
    ("A2", [math.log(2), 0, 0], 2),
    ("A3", [1, -2, 0.5], 0),
    ("A4", [0.3, -0.7], 1),
    ("A5", [0], 0),
]
SHARED_W = [[0.4, -0.2, 0.1], [0.1, 0.5, -0.4], [0.0, 0.0, 0.0]]
SHARED_X = [0, 1, 0, 0]
SHARED_Y = [1, 0, 2, 1]


def sample_a3():
    _, scores, target = SAMPLES[2]
    return torch.tensor(scores, dtype=DTYPE, requires_grad=True), target


def experiment_a(checks: Checks) -> dict:
    samples = {}
    for name, scores, target in SAMPLES:
        item = compare_sample(scores, target)
        samples[name] = item
        checks.entries.extend({**c, "name": f"{name}/{c['name']}"} for c in item["checks"])

    z = torch.tensor([math.log(p) for p in (0.2, 0.5, 0.3)], dtype=DTYPE)
    expected = reference(values(z), 0)["gradient"][0]
    scan = []
    for h in (1e-1, 1e-3, 1e-6, 1e-16):
        plus, minus = z.clone(), z.clone()
        plus[0] += h
        minus[0] -= h
        lp, lm = cross_entropy(plus, 0).item(), cross_entropy(minus, 0).item()
        estimate = (lp - lm) / (2 * h)
        if not math.isfinite(estimate):
            raise FloatingPointError("步长扫描的差分出现非有限值")
        plus_changed = plus[0].item() != z[0].item()
        minus_changed = minus[0].item() != z[0].item()
        error = abs(estimate - expected)
        if not plus_changed or not minus_changed:
            status = "perturbation_rounded_away"
            explanation = "至少一侧扰动未改变实际保存值，不能据此判断真实梯度为零。"
        elif error > DIFFERENCE_ATOL:
            status = "outside_standard_tolerance"
            explanation = "此h超出常规误差阈值；结合多种h观察近似及浮点误差。"
        else:
            status = "within_standard_tolerance"
            explanation = "此h在当前数值尺度下与解析梯度吻合。"
        scan.append({
            "h": h, "base_score": z[0].item(), "plus_score": plus[0].item(),
            "minus_score": minus[0].item(), "plus_changed": plus_changed,
            "minus_changed": minus_changed, "loss_plus": lp, "loss_minus": lm,
            "estimate": estimate, "absolute_error": error,
            "observation_status": status, "explanation": explanation,
        })
    return {"samples": samples, "step_size_scan": {
        "logits": values(z), "target": 0, "component": 0, "analytic_gradient": expected,
        "rows": scan, "note": "扫描记录数值失准，不要求每个h通过梯度精度检查。",
    }}


def experiment_b(checks: Checks) -> dict:
    z, target = sample_a3()
    lr = 0.1
    optimizer = make_sgd([z], lr)
    optimizer.zero_grad(set_to_none=True)
    states = {"before_forward": snapshot(z)}
    old = states["before_forward"]["parameter"]
    analytic = reference(old, target)["gradient"]
    loss = cross_entropy(z, target)
    loss_before = loss.item()
    loss.backward()
    states["after_backward"] = snapshot(z)
    optimizer.step()
    states["after_step"] = snapshot(z)
    with torch.no_grad():
        loss_after = cross_entropy(z, target).item()
    optimizer.zero_grad(set_to_none=True)
    states["after_clear"] = snapshot(z)
    optimizer.step()  # 没有新 backward，此处不应更新。
    states["step_without_backward"] = snapshot(z)
    expected_update = [v - lr * g for v, g in zip(old, analytic)]
    checks.truth("初始梯度None", states["before_forward"]["grad"] is None)
    checks.truth("backward保持参数", states["after_backward"]["parameter"] == old)
    checks.close("backward梯度", states["after_backward"]["grad"], analytic)
    checks.close("SGD解析更新", states["after_step"]["parameter"], expected_update)
    checks.truth("step不清梯度", states["after_step"]["grad"] == states["after_backward"]["grad"])
    checks.truth("清梯度保留参数", states["after_clear"]["parameter"] == states["after_step"]["parameter"])
    checks.truth("清后梯度None", states["after_clear"]["grad"] is None)
    checks.truth("无backward不更新", states["step_without_backward"] == states["after_clear"])
    checks.close("更新后重新评估", loss_after, reference(expected_update, target)["loss"])
    checks.truth("本例小步长损失下降", loss_after < loss_before)
    return {"target": target, "learning_rate": lr, "momentum": 0, "weight_decay": 0,
            "loss_before": loss_before, "loss_after": loss_after,
            "analytic_gradient": analytic, "expected_update": expected_update, "states": states}


def experiment_c(checks: Checks) -> dict:
    model = TrainableBigram(len(SHARED_W), dtype=DTYPE)
    with torch.no_grad():
        model.score_table.copy_(torch.tensor(SHARED_W, dtype=DTYPE))
    x = torch.tensor(SHARED_X, dtype=torch.long)
    y = torch.tensor(SHARED_Y, dtype=torch.long)
    batch_size = len(SHARED_X)
    parameter = model.score_table
    optimizer = make_sgd(model.parameters(), 0.2)
    optimizer.zero_grad(set_to_none=True)
    before = snapshot(parameter)
    analytic = [[0.0] * len(SHARED_W) for _ in SHARED_W]
    contributions = []
    for position, (input_id, target) in enumerate(zip(SHARED_X, SHARED_Y)):
        item = reference(SHARED_W[input_id], target)
        scaled = [g / batch_size for g in item["gradient"]]
        for j, g in enumerate(scaled):
            analytic[input_id][j] += g
        contributions.append({
            "position": position, "input_id": input_id, "target": target,
            "probabilities": item["probabilities"], "loss": item["loss"],
            "single_position_gradient": item["gradient"], "mean_scaled_contribution": scaled,
        })

    def batch_loss(table):
        result = F.cross_entropy(table[x], y, reduction="mean")
        ensure_finite(result, "批次损失")
        return result

    numerical = central_difference(parameter, batch_loss)
    checks.truth("差分保持整个模型", snapshot(parameter) == before)
    logits = model(x)
    loss = F.cross_entropy(logits, y, reduction="mean")
    ensure_finite(loss, "模型损失")
    loss.backward()
    after_backward = snapshot(parameter)
    checks.truth("logits形状B乘V", list(logits.shape) == [batch_size, len(SHARED_W)])
    checks.close("重复输入查相同行", values(logits[0]), values(logits[2]), 0)
    checks.close("重复输入第三次", values(logits[0]), values(logits[3]), 0)
    checks.close("平均损失独立参考", loss.item(), math.fsum(c["loss"] for c in contributions) / batch_size)
    checks.close("共享表解析与autograd", values(parameter.grad), analytic)
    checks.close("共享表解析与差分", values(numerical), analytic, DIFFERENCE_ATOL)
    checks.truth("backward不改表", after_backward["parameter"] == before["parameter"])
    expected = [[v - 0.2 * g for v, g in zip(row, grad)] for row, grad in zip(SHARED_W, analytic)]
    optimizer.step()
    after_step = snapshot(parameter)
    checks.close("共享表SGD", after_step["parameter"], expected)
    checks.close("未查询行零梯度", after_backward["grad"][2], [0, 0, 0], 0)
    checks.truth("未查询行保持参数", after_step["parameter"][2] == before["parameter"][2])
    return {
        "model_class": "learning.stage02_trainable_bigram.model.TrainableBigram",
        "x": SHARED_X, "y": SHARED_Y, "batch_size": batch_size, "vocab_size": len(SHARED_W),
        "reduction": "mean", "learning_rate": 0.2, "h": DEFAULT_H,
        "logits": values(logits), "loss": loss.item(), "contributions": contributions,
        "analytic_gradient": analytic, "difference_gradient": values(numerical),
        "autograd_gradient": after_backward["grad"], "expected_update": expected,
        "before": before, "after_backward": after_backward, "after_step": after_step,
    }


def experiment_d(checks: Checks) -> dict:
    branches = {}
    for clear in (False, True):
        name = "clear_between" if clear else "keep_previous"
        z, target = sample_a3()
        optimizer = make_sgd([z], 0.1)
        optimizer.zero_grad(set_to_none=True)
        initial = snapshot(z)
        gradient = reference(values(z), target)["gradient"]
        cross_entropy(z, target).backward()
        first = snapshot(z)
        if clear:
            optimizer.zero_grad(set_to_none=True)
        before_second = snapshot(z)
        second_loss = 2 * cross_entropy(z, target)  # 新的前向计算图。
        second_loss.backward()
        second = snapshot(z)
        new_contribution = [2 * g for g in gradient]
        old_contribution = [0.0] * len(gradient) if clear else gradient
        expected = [old + new for old, new in zip(old_contribution, new_contribution)]
        checks.close(f"{name}/第一轮贡献", first["grad"], gradient)
        checks.close(f"{name}/第二轮累计", second["grad"], expected)
        checks.truth(f"{name}/没有step参数不变", initial["parameter"] == first["parameter"] == second["parameter"])
        checks.truth(f"{name}/清除行为", (before_second["grad"] is None) if clear else before_second["grad"] == first["grad"])
        branches[name] = {
            "target": target, "second_loss_scale": 2, "initial": initial,
            "first_new_contribution": gradient, "after_first_backward": first,
            "before_second_backward": before_second, "second_new_contribution": new_contribution,
            "expected_final_gradient": expected, "after_second_backward": second,
        }
    return branches


def tensor_info(tensor):
    return {"value": values(tensor), "requires_grad": tensor.requires_grad,
            "grad_fn": None if tensor.grad_fn is None else type(tensor.grad_fn).__name__}


def experiment_e(checks: Checks) -> dict:
    cases = {}
    for case in ("logging", "detached_saved", "detached_used", "rewrapped", "no_grad_square", "no_grad_partial"):
        w = torch.tensor(1.5, dtype=DTYPE, requires_grad=True)
        saved = None
        logged_value = None
        if case.startswith("no_grad"):
            with torch.no_grad():
                a = 2 * w
        else:
            a = 2 * w
        if case == "detached_saved":
            saved = a.detach()
        elif case == "detached_used":
            a = a.detach()
        loss = a * w if case == "no_grad_partial" else a * a
        original_loss_info = tensor_info(loss)
        if case in ("logging", "rewrapped"):
            logged_value = loss.item()
        if case == "rewrapped":
            loss = torch.tensor(logged_value, dtype=DTYPE, requires_grad=True)

        expected_error = case in ("detached_used", "no_grad_square")
        if case in ("logging", "detached_saved"):
            expected_gradient = 8 * w.item()  # L=(2w)^2，数学求导。
        elif case == "no_grad_partial":
            expected_gradient = a.item()  # a视作常量，只走直接乘w的路径。
        else:
            expected_gradient = None
        error = None
        try:
            loss.backward()
        except RuntimeError as exc:
            # 只将没有可求导目标的反向调用识别为本实验预期错误。
            if not expected_error or loss.requires_grad or loss.grad_fn is not None:
                raise
            error = {"type": type(exc).__name__, "message": str(exc)}
        checks.truth(f"{case}/backward状态", (error is not None) == expected_error)
        checks.truth(f"{case}/目标是否可求导", loss.requires_grad == (not expected_error))
        if expected_gradient is None:
            checks.truth(f"{case}/w未收到梯度", w.grad is None)
        else:
            checks.truth(f"{case}/w收到梯度", w.grad is not None)
            if w.grad is not None:
                checks.close(f"{case}/路径解析梯度", values(w.grad), expected_gradient)
        if case == "rewrapped":
            checks.truth("rewrapped/新叶梯度存在", loss.grad is not None)
            if loss.grad is not None:
                checks.close("rewrapped/自身导数", values(loss.grad), 1.0)
        checks.close(f"{case}/中间值", a.item(), 3.0, 0)
        checks.close(f"{case}/目标数值", loss.item(), 4.5 if case == "no_grad_partial" else 9.0, 0)
        checks.close(f"{case}/w没有更新", w.item(), 1.5, 0)
        cases[case] = {
            "w": tensor_info(w), "a": tensor_info(a), "original_loss": original_loss_info,
            "backward_target": tensor_info(loss), "saved": None if saved is None else tensor_info(saved),
            "logged_value": logged_value, "w_grad": None if w.grad is None else values(w.grad),
            "new_leaf_grad": values(loss.grad) if case == "rewrapped" and loss.grad is not None else None,
            "expected_w_gradient": expected_gradient, "backward_succeeded": error is None,
            "outcome": "expected_error" if expected_error and error is not None else ("unexpected_outcome" if expected_error else "completed"),
            "error": error,
        }
    return cases


def experiment_f(checks: Checks) -> dict:
    runs = []
    for lr in (0.1, 0.75, 1.5):
        w = torch.tensor(2.0, dtype=DTYPE, requires_grad=True)
        optimizer = make_sgd([w], lr)
        optimizer.zero_grad(set_to_none=True)
        initial = snapshot(w)
        loss = w.square()
        loss.backward()
        after_backward = snapshot(w)
        expected_gradient = 2 * initial["parameter"]
        expected_parameter = initial["parameter"] - lr * expected_gradient
        optimizer.step()
        after_step = snapshot(w)
        with torch.no_grad():
            loss_after = w.square().item()
        checks.close(f"lr={lr}/梯度", after_backward["grad"], expected_gradient)
        checks.close(f"lr={lr}/参数", after_step["parameter"], expected_parameter)
        checks.close(f"lr={lr}/新损失", loss_after, expected_parameter ** 2)
        checks.truth(f"lr={lr}/backward不更新", initial["parameter"] == after_backward["parameter"])
        runs.append({
            "learning_rate": lr, "initial": initial, "loss_before": loss.item(),
            "after_backward": after_backward, "after_step": after_step,
            "analytic_gradient": expected_gradient, "expected_parameter": expected_parameter,
            "loss_after": loss_after, "loss_decreased": loss_after < loss.item(),
            "crossed_minimum": initial["parameter"] * after_step["parameter"] < 0,
        })
    return {"loss_function": "w**2", "momentum": 0, "weight_decay": 0, "runs": runs}


EXPERIMENTS = {"A": experiment_a, "B": experiment_b, "C": experiment_c,
               "D": experiment_d, "E": experiment_e, "F": experiment_f}
