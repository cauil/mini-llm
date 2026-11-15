"""从项目根目录运行：python -m learning.stage03_gradients.run。"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import traceback

import torch

from .core import Checks, DTYPE, compare_sample, make_sgd
from .experiments import EXPERIMENTS

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "results.json"


def validate_bad_inputs(checks: Checks) -> dict:
    cases = [
        ("empty_logits", [], 0, 1e-5),
        ("bool_score", [True, 0], 0, 1e-5),
        ("nested_scores", [[0]], 0, 1e-5),
        ("nan_score", [float("nan")], 0, 1e-5),
        ("inf_score", [float("inf")], 0, 1e-5),
        ("negative_target", [0, 1], -1, 1e-5),
        ("large_target", [0, 1], 2, 1e-5),
        ("float_target", [0, 1], 0.0, 1e-5),
        ("bool_target", [0, 1], True, 1e-5),
        ("string_target", [0, 1], "0", 1e-5),
    ]
    cases.extend((f"bad_h_{i}", [0, 1], 0, h) for i, h in enumerate(
        [0, -1, float("inf"), float("nan"), True, "0.1"]))
    records = []
    for name, logits, target, h in cases:
        error = None
        try:
            compare_sample(logits, target, h)
        except ValueError as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        checks.truth(name, error is not None)
        records.append({"name": name, "input_repr": repr({"logits": logits, "target": target, "h": h}),
                        "outcome": "expected_rejection" if error else "unexpected_acceptance", "error": error})
    for i, lr in enumerate([0, -1, float("inf"), float("nan"), True, "0.1"]):
        error = None
        try:
            make_sgd([torch.tensor(1.0, dtype=DTYPE, requires_grad=True)], lr)
        except ValueError as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        name = f"bad_learning_rate_{i}"
        checks.truth(name, error is not None)
        records.append({"name": name, "input_repr": repr(lr),
                        "outcome": "expected_rejection" if error else "unexpected_acceptance", "error": error})
    return {"rejections": records}


def refresh_summary(report: dict) -> None:
    checks = Checks(report["checks"])
    report["summary"] = checks.summary()
    report["summary"]["status"] = "passed" if report["summary"]["failed"] == 0 else "failed"


def build_report() -> dict:
    report = {
        "schema_version": 1, "task_version": "v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "pytorch": torch.__version__,
                        "device": "cpu", "dtype": str(DTYPE)},
        "task": "tasks/阶段3_梯度与更新实验任务书.md",
        "experiments": {}, "checks": [],
    }
    groups = {**EXPERIMENTS, "input_validation": validate_bad_inputs}
    for name, experiment in groups.items():
        checks = Checks()
        try:
            observation = experiment(checks)
        except Exception as exc:
            # 非预期错误记录为失败，保留诊断并继续收集其他独立实验。
            observation = {"unexpected_error": {
                "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}
            checks.truth("运行无非预期异常", False, f"{type(exc).__name__}: {exc}")
        report["experiments"][name] = {
            "observations": observation, "checks": checks.entries, "summary": checks.summary()}
        report["checks"].extend({**entry, "name": f"{name}/{entry['name']}"} for entry in checks.entries)
    refresh_summary(report)
    return report


def save_report(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    def write():
        # 禁止把 NaN/Inf 当成合法 JSON 输出；任何保存问题让主入口失败。
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    write()
    matched = json.loads(output.read_text(encoding="utf-8")) == report
    report["checks"].append({"name": "output/json_roundtrip", "passed": matched,
                              "detail": "整份结构与所有数值保存后重读核对"})
    refresh_summary(report)
    write()
    if json.loads(output.read_text(encoding="utf-8")) != report:
        raise OSError("最终JSON读回与内存结果不一致")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="运行阶段3六组实验并保存真实检查结果")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="结果JSON路径")
    args = parser.parse_args(argv)
    report = build_report()
    try:
        save_report(report, args.output)
    except (OSError, ValueError, TypeError) as exc:
        print(f"结果保存失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    summary = report["summary"]
    print(f"阶段3检查：{summary['passed']}/{summary['total']} 通过；失败 {summary['failed']} 项")
    print(f"结果：{args.output.resolve()}")
    if summary["failed_names"]:
        print("失败项：" + ", ".join(summary["failed_names"]), file=sys.stderr)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
