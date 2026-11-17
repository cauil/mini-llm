"""记录独立参考、输入边界、形状及训练路径诊断的真实检查证据。"""

import copy
import math

import torch
from torch.nn import functional as F

from learning.stage00_char_codec.tokenizer import CharTokenizer
from .data import prepare_data
from .model import ContextMLP, ModelConfig, generate, snapshot
from .reference import reference
from .training import evaluate, make_optimizer, train


class Checks:
    def __init__(self):
        self.items = []
        self.experiments = {}

    def record(self, name, passed, **details):
        self.items.append({"name": name, "passed": bool(passed), **details})

    def close(self, name, actual, expected, *, atol=1e-8, rtol=1e-6):
        a = torch.as_tensor(actual, dtype=torch.float64)
        b = torch.as_tensor(expected, dtype=torch.float64)
        same_shape = a.shape == b.shape
        error = float((a - b).abs().max()) if same_shape and a.numel() else None
        self.record(name, same_shape and torch.allclose(a, b, atol=atol, rtol=rtol),
                    max_absolute_error=error, actual_shape=list(a.shape),
                    expected_shape=list(b.shape), atol=atol, rtol=rtol)

    def rejects(self, name, function, expected=(ValueError,)):
        try:
            function()
        except expected as error:
            self.record(name, True, expected_exception=type(error).__name__, message=str(error))
        except Exception as error:
            self.record(name, False, unexpected_exception=type(error).__name__, message=str(error))
        else:
            self.record(name, False, message="非法输入未拒绝")

    def result(self):
        passed = sum(item["passed"] for item in self.items)
        return {"count": len(self.items), "passed": passed,
                "failed": len(self.items) - passed, "all_passed": passed == len(self.items),
                "checks": self.items, "experiments": self.experiments}


def fixed_cases():
    return [
        {"name": "single", "config": ModelConfig(3, 2, 1, 2),
         "x": [[0, 2]], "y": [2],
         "parameters": {"embedding.weight": [[1.0], [-1.0], [2.0]],
                        "layer1.weight": [[1.0, 1.0], [1.0, -1.0]], "layer1.bias": [0., 0.],
                        "layer2.weight": [[0., 1.], [math.log(2) / 3, -1.], [0., 2.]],
                        "layer2.bias": [0., 0., 0.]}},
        {"name": "batch_repeated", "config": ModelConfig(5, 3, 2, 3),
         "x": [[0, 1, 0], [2, 0, 1], [1, 1, 2], [0, 2, 0]], "y": [1, 2, 0, 1],
         "parameters": {"embedding.weight": [[.3, -.2], [-.4, .7], [.8, .1], [.2, .5], [-.6, .4]],
                        "layer1.weight": [[.2, -.3, .4, .1, -.2, .5],
                                          [-.4, .2, .3, -.5, .1, .2],
                                          [.6, .1, -.2, .4, .3, -.1]],
                        "layer1.bias": [.11, -.23, .17],
                        "layer2.weight": [[.2, -.4, .1], [-.3, .5, .2], [.4, .1, -.2],
                                          [-.1, -.3, .6], [.5, -.2, -.4]],
                        "layer2.bias": [.1, -.2, .05, .3, -.1]}}
    ]


def make_case(case):
    model = ContextMLP(case["config"])
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(torch.tensor(case["parameters"][name], dtype=torch.float64))
    return model, torch.tensor(case["x"], dtype=torch.long), torch.tensor(case["y"], dtype=torch.long)


def gradient_case(checks: Checks, case: dict) -> dict:
    model, x, y = make_case(case)
    model.zero_grad(set_to_none=True)
    features = model.features(x, retain=True)
    loss = F.cross_entropy(features["logits"], y)
    loss.backward()
    ref = reference(case["parameters"], case["x"], case["y"])
    name = case["name"]
    a = features["a"].detach()
    checks.record(f"{name}/relu_away_from_zero", bool((a.abs() > 1e-6).all()))
    checks.record(f"{name}/relu_both_branches", bool((a > 0).any() and (a < 0).any()))
    for key, tensor in features.items():
        checks.close(f"{name}/forward/{key}", tensor.detach(), ref["forward"][key])
    checks.close(f"{name}/loss", loss.detach(), ref["mean_loss"])
    checks.close(f"{name}/probabilities", features["logits"].detach().softmax(-1), ref["probabilities"])
    actual = snapshot(model, gradients=True)
    actual.update({key: value.grad.tolist() for key, value in features.items()})
    for key, expected in ref["gradients"].items():
        checks.close(f"{name}/gradient/{key}", actual[key], expected)
    for token in set(range(model.config.vocab_size)) - set(x.flatten().tolist()):
        checks.close(f"{name}/unused_embedding/{token}", model.embedding.weight.grad[token],
                     torch.zeros(model.config.embedding_dim))
    # 额外确认损失定义改变时，所有参数梯度按比例缩放。
    model.zero_grad(set_to_none=True)
    (2 * F.cross_entropy(model(x), y)).backward()
    for key, p in model.named_parameters():
        checks.close(f"{name}/scaled_loss/{key}", p.grad,
                     2 * torch.tensor(ref["gradients"][key], dtype=torch.float64))
    return {"parameters": case["parameters"], "x": case["x"], "y": case["y"],
            "actual_forward": {key: value.detach().tolist() for key, value in features.items()},
            "actual_probabilities": features["logits"].detach().softmax(-1).tolist(),
            "actual_sample_losses": F.cross_entropy(features["logits"].detach(), y, reduction="none").tolist(),
            "actual_mean_loss": float(loss.detach()), "reference": ref, "autograd": actual}


def shape_checks(checks):
    configurations = [(3, 1, 1, 2, 1), (5, 2, 3, 4, 4), (7, 3, 2, 5, 5)]
    for V, k, D, H, B in configurations:
        model = ContextMLP(ModelConfig(V, k, D, H))
        x = torch.tensor([[(r + p) % V for p in range(k)] for r in range(B)])
        x[0] = 0  # 同时覆盖重复编号。
        prefix = f"shape/V{V}_k{k}_D{D}_H{H}_B{B}"
        features = model.features(x)
        for key, shape in (("emb", [B, k, D]), ("z", [B, k * D]),
                           ("a", [B, H]), ("h", [B, H]), ("logits", [B, V])):
            checks.record(f"{prefix}/{key}", list(features[key].shape) == shape)
        one_hot = F.one_hot(x, V).to(torch.float64)
        checks.close(f"{prefix}/one_hot", features["emb"].detach(), one_hot @ model.embedding.weight.detach())
        individual = torch.cat([model(row[None]) for row in x])
        checks.close(f"{prefix}/batch_equals_individual", features["logits"].detach(), individual.detach())
        checks.close(f"{prefix}/permutation", model(x.flip(0)).detach(), features["logits"].detach().flip(0))
        reversed_z = model.features(x.flip(1))["z"].detach()
        checks.close(f"{prefix}/position_order", reversed_z,
                     features["emb"].detach().flip(1).reshape(B, k * D))
        checks.close(f"{prefix}/bias_broadcast", features["a"].detach(),
                     features["z"].detach() @ model.layer1.weight.detach().T + model.layer1.bias.detach())
        expected_count = V * D + H * k * D + H + V * H + V
        checks.record(f"{prefix}/parameter_count", model.parameter_info()["total_trainable"] == expected_count)


def data_and_boundary_checks(checks):
    tokenizer = CharTokenizer({"春": 0, "风": 1, "雨": 2})
    raw = {"train": [{"id": "a", "text": "春风雨春"}, {"id": "short", "text": "风春"}],
           "val": [{"id": "b", "text": "雨风春风"}, {"id": "empty", "text": ""}]}
    groups, metadata = prepare_data(raw, tokenizer, 2)
    checks.record("data/manual_train", groups["train"].x.tolist() == [[0, 1], [1, 2]]
                  and groups["train"].y.tolist() == [2, 0])
    checks.record("data/manual_val", groups["val"].x.tolist() == [[2, 1], [1, 0]]
                  and groups["val"].y.tolist() == [0, 1])
    by_id = {doc["id"]: doc["text"] for docs in raw.values() for doc in docs}
    for split, data in groups.items():
        for i, sample in enumerate(data.samples):
            text = by_id[sample["doc_id"]]
            start = sample["start"]
            checks.record(f"data/source/{split}/{i}",
                          sample["split"] == split and sample["context_text"] == text[start:start + 2]
                          and sample["target_text"] == text[start + 2]
                          and sample["target_position"] == start + 2)
    checks.record("data/zero_sample_documents", {d["doc_id"] for d in metadata["documents"]
                  if d["sample_count"] == 0} == {"short", "empty"})
    separated = {"train": [{"id": "a", "text": "春风"}, {"id": "b", "text": "雨春"}],
                 "val": [{"id": "v", "text": "风雨"}]}
    empty, _ = prepare_data(separated, tokenizer, 2)
    model = ContextMLP(ModelConfig(3, 2, 2, 4))
    checks.record("data/no_cross_document", all(len(d.y) == 0 for d in empty.values()))
    checks.record("data/empty_eval", evaluate(model, empty["val"].x, empty["val"].y)["loss"] is None)
    checks.rejects("data/empty_train", lambda: train(model, empty["train"].x, empty["train"].y))
    for field in ("vocab_size", "context_length", "embedding_dim", "hidden_dim"):
        for invalid in (0, -1, True, 1.5):
            arguments = {"vocab_size": 3, field: invalid}
            checks.rejects(f"invalid/{field}/{invalid!r}", lambda a=arguments: ModelConfig(**a))
    for name, invalid in {
        "rank": torch.tensor([0, 1]), "length": torch.tensor([[0]]),
        "float": torch.tensor([[0., 1.]]), "empty": torch.empty((0, 2), dtype=torch.long),
        "negative": torch.tensor([[-1, 1]]), "upper": torch.tensor([[0, 3]])
    }.items():
        checks.rejects(f"invalid/input/{name}", lambda value=invalid: model(value))
    for value in (0, -1, float("nan"), float("inf"), True):
        checks.rejects(f"invalid/lr/{value}", lambda value=value: make_optimizer(model, value))
    for value in (-1, True, 1.5):
        checks.rejects(f"invalid/steps/{value}", lambda value=value:
                       train(model, groups["train"].x, groups["train"].y, steps=value))
    checks.rejects("data/unknown_even_short", lambda: prepare_data(
        {"train": [{"id": "unknown", "text": "猫"}], "val": []}, tokenizer, 2), (KeyError,))
    checks.rejects("data/duplicate_id", lambda: prepare_data(
        {"train": [{"id": "a", "text": "春风雨"}],
         "val": [{"id": "a", "text": "雨风春"}]}, tokenizer, 2))
    for name, kwargs in {
        "short": {"prompt": "春"}, "unknown": {"prompt": "春猫"},
        "negative_length": {"num_new_tokens": -1}, "method": {"method": "bad"}
    }.items():
        options = {"prompt": "春风", "num_new_tokens": 2, **kwargs}
        checks.rejects(f"generation/{name}", lambda options=options: generate(model, tokenizer, **options))
    checks.record("generation/zero_length", generate(model, tokenizer, "春风", 0)["text"] == "春风")
    with torch.no_grad():
        model.layer1.weight[0, 0] = float("nan")
    checks.rejects("nonfinite/parameter", lambda: model(groups["train"].x), (FloatingPointError,))


def diagnostic_checks(checks):
    case = fixed_cases()[0]
    variants = {}
    for detached in (False, True):
        model, x, y = make_case(case)
        f = model.features(x, detach_hidden=detached)
        loss = F.cross_entropy(f["logits"], y)
        loss.backward()
        variants["detached" if detached else "connected"] = {
            "h": f["h"].detach().tolist(), "h_requires_grad": f["h"].requires_grad,
            "logits": f["logits"].detach().tolist(), "loss": float(loss.detach()),
            "gradients": snapshot(model, gradients=True)}
    good, detached = variants["connected"], variants["detached"]
    checks.close("detach/same_logits", detached["logits"], good["logits"])
    checks.close("detach/same_loss", detached["loss"], good["loss"])
    for name in ("embedding.weight", "layer1.weight", "layer1.bias"):
        checks.record(f"detach/disconnected/{name}", detached["gradients"][name] is None
                      and good["gradients"][name] is not None)
    for name in ("layer2.weight", "layer2.bias"):
        checks.close(f"detach/downstream/{name}", detached["gradients"][name], good["gradients"][name])
        checks.record(f"detach/nonzero/{name}", bool(torch.tensor(detached["gradients"][name]).abs().sum() > 0))
    variants["explanation"] = "detach 不改变前向值，阻断到输入来源的路径；后续参数仍能求导。"
    checks.experiments["detach"] = variants

    variants = {}
    original = ContextMLP(ModelConfig(3, 2, 2, 16, 42))
    for zero in (False, True):
        model = copy.deepcopy(original)
        if zero:
            with torch.no_grad():
                for layer in (model.layer1, model.layer2):
                    layer.weight.zero_()
                    layer.bias.zero_()
        x, y = torch.tensor([[0, 2]]), torch.tensor([2])
        features = model.features(x)
        loss = F.cross_entropy(features["logits"], y)
        loss.backward()
        variants["zero_linear" if zero else "random_weights_zero_bias"] = {
            "parameters": snapshot(model), "h": features["h"].detach().tolist(),
            "logits": features["logits"].detach().tolist(), "loss": float(loss.detach()),
            "gradients": snapshot(model, gradients=True)}
    random_case, zero_case = variants["random_weights_zero_bias"], variants["zero_linear"]
    for name, grad in zero_case["gradients"].items():
        checks.record(f"initialization/zero/{name}", grad is not None and
                      (bool(torch.tensor(grad).abs().sum() > 0) if name == "layer2.bias"
                       else bool(torch.tensor(grad).eq(0).all())))
    checks.record("initialization/random_hidden_active", bool(torch.tensor(random_case["h"]).ne(0).any()))
    checks.record("initialization/random_embedding_gradient",
                  bool(torch.tensor(random_case["gradients"]["embedding.weight"]).ne(0).any()))
    variants["explanation"] = "两层全零令 h=0，输出权重梯度为0，输入路径也被零权重阻断；输出偏置仍可更新。"
    checks.experiments["initialization"] = variants


def run_checks() -> Checks:
    checks = Checks()
    shape_checks(checks)
    data_and_boundary_checks(checks)
    checks.experiments["gradients"] = [gradient_case(checks, case) for case in fixed_cases()]
    diagnostic_checks(checks)
    return checks
