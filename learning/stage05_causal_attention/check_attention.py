"""任务v1自检与可复现JSON；所有条件均依据本次实际运算。"""

import argparse
from dataclasses import asdict, fields
import json
import math
from pathlib import Path
import platform

import torch

from .model import AttentionConfig, AttentionTrace, SingleHeadCausalAttention, causal_attention
from .reference import attention_reference, project_reference
from .diagnostics import broken_attention

ATOL, RTOL = 1e-8, 1e-6
DTYPE = torch.float64


def tensor(value):
    return torch.tensor(value, dtype=DTYPE)


def serializable(value):
    """只在报告边界脱图；严格JSON用字符串标识非有限诊断值。"""
    if isinstance(value, torch.Tensor):
        return serializable(value.detach().cpu().tolist())
    if isinstance(value, AttentionTrace):
        return {f.name: serializable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("-inf" if value < 0 else "inf")
    return value


class Checks:
    def __init__(self):
        self.items = []

    def require(self, name, condition, **details):
        self.items.append({"name": name, "passed": bool(condition), **serializable(details)})

    def close(self, name, actual, expected):
        compatible = actual.shape == expected.shape
        valid = compatible and bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
        error = float((actual - expected).abs().max()) if valid else None
        passed = valid and torch.allclose(actual, expected, atol=ATOL, rtol=RTOL)
        self.require(name, passed, max_abs_error=error,
                     actual_shape=list(actual.shape), expected_shape=list(expected.shape))

    def rejects(self, name, operation, errors=(ValueError, TypeError, FloatingPointError)):
        try:
            operation()
        except errors as error:
            self.require(name, True, exception=type(error).__name__, message=str(error))
        else:
            self.require(name, False, message="非法输入未被拒绝")


def weights_checks(checks, name, output, weights):
    length = weights.shape[-1]
    future = torch.ones(length, length, dtype=torch.bool).triu(1)
    checks.close(name + "/row_sum", weights.sum(-1), torch.ones_like(weights.sum(-1)))
    checks.require(name + "/future_zero", bool((weights[:, future] == 0).all()))
    checks.require(name + "/finite", bool(torch.isfinite(weights).all() and torch.isfinite(output).all()))


def parameters(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def classroom(checks):
    model = SingleHeadCausalAttention(AttentionConfig(2, 1, 2))
    x = tensor([[[1, 0], [0, 1], [1, 1]]])
    with torch.no_grad():
        model.query.weight.copy_(tensor([[1, 1]]))
        model.key.weight.copy_(tensor([[0, math.log(2)]]))
        model.value.weight.copy_(torch.eye(2, dtype=DTYPE))
        trace = model.inspect(x)
        expected = {
            "q": tensor([[[1], [1], [2]]]),
            "k": tensor([[[0], [math.log(2)], [math.log(2)]]]),
            "v": x.clone(),
            "scores": tensor([[[0, math.log(2), math.log(2)],
                               [0, math.log(2), math.log(2)],
                               [0, 2 * math.log(2), 2 * math.log(2)]]]),
            "weights": tensor([[[1, 0, 0], [1/3, 2/3, 0], [1/9, 4/9, 4/9]]]),
            "output": tensor([[[1, 0], [1/3, 2/3], [5/9, 8/9]]]),
        }
        for name, value in expected.items():
            checks.close("classroom/" + name, getattr(trace, name), value)
        expected_mask = torch.tensor([[False, True, True], [False, False, True], [False, False, False]])
        checks.require("classroom/mask", torch.equal(trace.future_mask, expected_mask))
        checks.require("classroom/masked_negative_infinity",
                       bool(torch.isneginf(trace.masked_scores[:, expected_mask]).all()))
        checks.close("classroom/allowed_scores", trace.masked_scores[:, ~expected_mask],
                     trace.scores[:, ~expected_mask])
        weights_checks(checks, "classroom", trace.output, trace.weights)
        return serializable({"x": x, "config": asdict(model.config), "parameters_pytorch_layout": parameters(model),
                             "trace": trace, "expected": expected})


@torch.no_grad()
def reference_cases(checks):
    cases = []
    for index, (b, t, d, dk, dv) in enumerate(((1, 1, 3, 2, 4), (2, 4, 5, 3, 2), (2, 5, 4, 2, 3))):
        seed = 100 + index
        model = SingleHeadCausalAttention(AttentionConfig(d, dk, dv, seed))
        x = torch.randn(b, t, d, dtype=DTYPE, generator=torch.Generator().manual_seed(seed + 100))
        before_x, before_params = x.clone(), parameters(model)
        trace = model.inspect(x)
        rq, rk, rv = [project_reference(x, layer.weight) for layer in (model.query, model.key, model.value)]
        expected_h, expected_a = attention_reference(rq, rk, rv)
        prefix = f"reference/{index}"
        for name, actual, expected in (("q", trace.q, rq), ("k", trace.k, rk), ("v", trace.v, rv),
                                        ("output", trace.output, expected_h), ("weights", trace.weights, expected_a)):
            checks.close(prefix + "/" + name, actual, expected)
        weights_checks(checks, prefix, trace.output, trace.weights)
        checks.close(prefix + "/forward", model(x), trace.output)
        core_h, core_a = causal_attention(trace.q, trace.k, trace.v)
        checks.close(prefix + "/public_core_h", core_h, trace.output)
        checks.close(prefix + "/public_core_a", core_a, trace.weights)
        checks.close(prefix + "/samplewise", torch.cat([model(x[i:i+1]) for i in range(b)]), trace.output)
        checks.require(prefix + "/input_unchanged", torch.equal(x, before_x))
        checks.require(prefix + "/parameters_unchanged", all(torch.equal(p, before_params[n]) for n, p in model.named_parameters()))
        if t == 1:
            checks.close(prefix + "/single_value", trace.output, trace.v)
            checks.require(prefix + "/single_weight", bool((trace.weights == 1).all()))
        if b > 1:
            changed = x.clone()
            changed[0] += 0.5
            checks.close(prefix + "/batch_isolation", model(changed)[1:], trace.output[1:])
        cases.append(serializable({"shape_B_T_D_dk_dv": [b, t, d, dk, dv], "config": asdict(model.config),
                                   "input_seed": seed + 100, "x": x, "parameters": before_params,
                                   "actual": trace, "reference_h": expected_h, "reference_a": expected_a}))
    model = SingleHeadCausalAttention(AttentionConfig(4, 2, 3, 200))
    before = parameters(model)
    parameter_ids = [id(p) for p in model.parameters()]
    variable = []
    for t in (1, 3, 5):
        x = torch.randn(2, t, 4, dtype=DTYPE, generator=torch.Generator().manual_seed(300 + t))
        trace = model.inspect(x)
        rh, ra = attention_reference(*model.project(x))
        checks.close(f"variable_t/{t}/h", trace.output, rh)
        checks.close(f"variable_t/{t}/a", trace.weights, ra)
        checks.require(f"variable_t/{t}/same_parameters", parameter_ids == [id(p) for p in model.parameters()]
                       and all(torch.equal(p, before[n]) for n, p in model.named_parameters()))
        variable.append(serializable({"t": t, "input_seed": 300+t, "x": x, "h": trace.output}))
    cases.append({"variable_length": variable, "config": asdict(model.config), "parameters": serializable(before)})
    return cases


def causal_fixture():
    model = SingleHeadCausalAttention(AttentionConfig(3, 2, 2, 7))
    with torch.no_grad():
        model.query.weight.copy_(tensor([[1, 0, 0], [0, 1, 0]]))
        model.key.weight.copy_(tensor([[0, 1, 0], [1, 0, 0.5]]))
        model.value.weight.copy_(tensor([[1, 0, 1], [0, 1, -1]]))
    x = tensor([[[0.2, 0.6, -0.1], [0.7, -0.3, 0.4], [-0.5, 0.8, 0.2], [0.4, 0.1, -0.6]],
                [[-0.3, 0.4, 0.5], [0.6, 0.2, -0.4], [0.1, -0.7, 0.3], [0.8, 0.5, -0.2]]])
    return model, x


def prefix_probe(operation, x, changed, cutoff):
    """所有正常/故障路径共用同一个实际输入扰动判定。cutoff为0基。"""
    before, a_before = operation(x)
    after, a_after = operation(changed)
    finite = all(bool(torch.isfinite(t).all()) for t in (before, after, a_before, a_after))
    prefix_before, prefix_after = before[:, :cutoff+1], after[:, :cutoff+1]
    causal = finite and torch.allclose(prefix_before, prefix_after, atol=ATOL, rtol=RTOL)
    row_sum_ok = finite and all(torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=ATOL, rtol=RTOL)
                               for a in (a_before, a_after))
    return {"cutoff_zero_based": cutoff, "finite": finite, "causal": bool(causal),
            "row_sum_ok": bool(row_sum_ok), "prefix_max_abs_error": float((prefix_before-prefix_after).abs().max()),
            "before_h": before, "after_h": after, "before_a": a_before, "after_a": a_after}


@torch.no_grad()
def causality_cases(checks):
    model, x = causal_fixture()
    original = model.inspect(x)
    checks.require("causal/nonconstant_values", not torch.allclose(original.v[:, 0], original.v[:, 1]))
    checks.require("causal/nonuniform_weights", not torch.allclose(original.weights[:, 1, :2],
                                                                   torch.full((2, 2), 0.5, dtype=DTYPE)))
    def correct(inputs):
        trace = model.inspect(inputs)
        return trace.output, trace.weights

    cases = []
    for cutoff in (0, 1):
        changed = x.clone()
        changed[:, cutoff+1:] += tensor([0.7, -0.4, 0.9])
        changed_trace = model.inspect(changed)
        checks.require(f"causal/{cutoff}/prefix_input_unchanged", torch.equal(x[:, :cutoff+1], changed[:, :cutoff+1]))
        checks.require(f"causal/{cutoff}/future_projection_changed", any(
            not torch.allclose(getattr(original, name)[:, cutoff+1:], getattr(changed_trace, name)[:, cutoff+1:])
            for name in ("q", "k", "v")))
        good = prefix_probe(correct, x, changed, cutoff)
        checks.require(f"causal/{cutoff}/correct", good["causal"] and good["finite"] and good["row_sum_ok"],
                       prefix_max_abs_error=good["prefix_max_abs_error"])
        weights_checks(checks, f"causal/{cutoff}", changed_trace.output, changed_trace.weights)
        faults = {}
        for fault in ("no_mask", "reversed_mask", "query_softmax"):
            bad = prefix_probe(lambda inputs: broken_attention(model, inputs, fault=fault), x, changed, cutoff)
            checks.require(f"causal/{cutoff}/{fault}/detected", bad["finite"] and not bad["causal"],
                           prefix_max_abs_error=bad["prefix_max_abs_error"], expected="有限输出下被因果性检查拒绝")
            if fault == "query_softmax":
                checks.require(f"causal/{cutoff}/{fault}/row_sum_detected", not bad["row_sum_ok"])
                mask = torch.ones(4, 4, dtype=torch.bool).triu(1)
                checks.require(f"causal/{cutoff}/{fault}/future_still_zero", bool((bad["after_a"][:, mask] == 0).all()))
            faults[fault] = serializable(bad)
        cases.append(serializable({"original_x": x, "changed_x": changed, "correct": good, "faults": faults}))
    return {"config": asdict(model.config), "parameters": serializable(parameters(model)), "cases": cases}


def backward_case(checks):
    model, x = causal_fixture()
    x.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    trace = model.inspect(x)
    r = tensor([[[0.1, -0.3], [0.5, 0.2], [-0.7, 0.6], [0.9, -0.4]],
                [[0.2, 0.8], [-0.5, 0.3], [0.4, -0.9], [0.7, 0.1]]])
    loss = (trace.output * r).sum()
    loss.backward()
    grads = {}
    for name, leaf in [("x", x), *model.named_parameters()]:
        grad = leaf.grad
        checks.require("backward/" + name, grad is not None and grad.shape == leaf.shape
                       and bool(torch.isfinite(grad).all()) and bool((grad != 0).any()))
        grads[name] = None if grad is None else grad.detach().clone()
    checks.require("parameters/registered", set(dict(model.named_parameters())) ==
                   {"query.weight", "key.weight", "value.weight"} and all(p.requires_grad for p in model.parameters()))
    count = sum(p.numel() for p in model.parameters())
    checks.require("parameters/count", count == model.config.d_model * (2 * model.config.d_key + model.config.d_value), count=count)
    return serializable({"x": x, "r": r, "loss": loss, "parameters": parameters(model), "gradients": grads,
                         "scope": "只验证反向连通性，不证明完整attention梯度正确性；没有optimizer或step"})


def boundary_cases(checks):
    for field in ("d_model", "d_key", "d_value"):
        for bad in (0, -1, True, 1.5):
            args = dict(d_model=3, d_key=2, d_value=2)
            args[field] = bad
            checks.rejects(f"boundary/config/{field}/{bad}", lambda args=args: AttentionConfig(**args))
    for seed in (-1, True, 1.5, 2**63):
        checks.rejects(f"boundary/seed/{seed}", lambda seed=seed: AttentionConfig(3, 2, 2, seed))
    model = SingleHeadCausalAttention(AttentionConfig(3, 2, 2))
    base = torch.ones(2, 4, 3, dtype=DTYPE)
    inputs = {"not_tensor": [], "rank": base[0], "width": base[:, :, :2], "empty_batch": base[:0],
              "empty_time": base[:, :0], "empty_feature": base[:, :, :0], "integer": base.long(),
              "dtype_mismatch": base.float(), "complex": base.to(torch.complex128)}
    for name, value in inputs.items():
        checks.rejects("boundary/model/" + name, lambda value=value: model(value))
    for bad in (float("nan"), float("inf"), -float("inf")):
        value = base.clone()
        value[0, 0, 0] = bad
        checks.rejects(f"boundary/model/nonfinite/{bad}", lambda value=value: model(value))
    q = torch.ones(2, 4, 2, dtype=DTYPE)
    cases = {"q_rank": (q[0], q, q), "q_integer": (q.long(), q, q), "q_empty_feature": (q[..., :0], q[..., :0], q),
             "batch_mismatch": (q, q[:1], q), "key_width": (q, q[..., :1], q), "v_time": (q, q, q[:, :3]),
             "v_batch": (q, q, q[:1]), "k_dtype": (q, q.float(), q), "v_integer": (q, q, q.long()),
             "empty": (q[:, :0], q[:, :0], q[:, :0]), "v_empty_feature": (q, q, q[..., :0])}
    for name, args in cases.items():
        checks.rejects("boundary/core/" + name, lambda args=args: causal_attention(*args))
    for index, name in enumerate(("q", "k", "v")):
        args = [q.clone() for _ in range(3)]
        args[index][0, 0, 0] = float("nan")
        checks.rejects("boundary/core/nonfinite_" + name, lambda args=args: causal_attention(*args))
    enormous = torch.full((1, 2, 2), 1e308, dtype=DTYPE)
    checks.rejects("boundary/core/score_overflow", lambda: causal_attention(enormous, enormous, torch.ones_like(enormous)))
    with torch.no_grad():
        model.query.weight[0, 0] = float("inf")
    checks.rejects("boundary/nonfinite_parameter", lambda: model(base))


def run_suite():
    checks = Checks()
    result = {"version": 1, "environment": {"python": platform.python_version(), "torch": torch.__version__,
               "device": "cpu", "dtype": "float64", "threads": torch.get_num_threads()},
              "tolerance": {"atol": ATOL, "rtol": RTOL}, "sections": {}}
    example = None
    for name, operation in (("classroom", classroom), ("reference", reference_cases),
                             ("causality", causality_cases), ("backward", backward_case), ("boundary", boundary_cases)):
        try:
            section = operation(checks)
            if name == "classroom":
                example = section
            else:
                result["sections"][name] = section
        except Exception as error:
            checks.require(name + "/unexpected_exception", False, exception=type(error).__name__, message=str(error))
    result["checks"] = checks.items
    failed = sum(not item["passed"] for item in checks.items)
    result["summary"] = {"total": len(checks.items), "passed": len(checks.items) - failed, "failed": failed}
    return serializable(result), example


def write_json(path, value):
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="阶段5检查；失败时非零退出，保存真实结果")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    result, example = run_suite()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "checks.json", result)
    write_json(args.output_dir / "example.json", example)
    print(json.dumps(result["summary"], ensure_ascii=False))
    for item in result["checks"]:
        if not item["passed"]:
            print("失败：", item["name"], item)
    return 1 if result["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
