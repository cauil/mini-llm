"""一条命令运行数据适配、训练、梯度检查、合成对照、生成及保存重载。"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import torch

from learning.stage00_char_codec.tokenizer import CharTokenizer
from learning.stage02_trainable_bigram.model import TrainableBigram
from .checks import Checks, run_checks
from .data import prepare_data
from .model import ContextMLP, ModelConfig, generate, load_checkpoint, save_checkpoint, snapshot
from .training import evaluate, train


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "learning/stage00_dataset"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"


def write_json(path: Path, value: dict) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def check_first_step(checks: Checks, prefix: str, record: dict) -> None:
    first = record["first_step"]
    if first is None:
        checks.record(f"{prefix}/zero_steps", record["steps"] == 0)
        return
    lr = record["learning_rate"]
    for name in first["before"]:
        before = torch.tensor(first["before"][name], dtype=torch.float64)
        grad = torch.tensor(first["gradients"][name], dtype=torch.float64)
        checks.close(f"{prefix}/backward_preserves/{name}", first["after_backward"][name], before,
                     atol=0, rtol=0)
        checks.close(f"{prefix}/sgd/{name}", first["after_step"][name], before - lr * grad,
                     atol=1e-12, rtol=1e-12)


def synthetic_experiment(checks: Checks) -> dict:
    tokenizer = CharTokenizer({"我": 0, "你": 1, "爱": 2, "猫": 3, "狗": 4})
    raw = {"train": [{"id": "synthetic_me", "text": "我爱猫"},
                     {"id": "synthetic_you", "text": "你爱狗"}], "val": []}
    groups, metadata = prepare_data(raw, tokenizer, 2)
    data = groups["train"]
    attempts = []
    # 有界、可追踪的重试；每次从同一初值重建，而不是把累计步数藏起来。
    for steps in (1000, 3000):
        config = ModelConfig(5, 2, 4, 16, 42)
        model = ContextMLP(config)
        record = train(model, data.x, data.y, steps=steps)
        with torch.no_grad():
            probabilities = model(data.x).softmax(-1)
        success = (bool((probabilities.argmax(-1) == data.y).all())
                   and bool((probabilities[torch.arange(len(data.y)), data.y] >= .9).all()))
        attempts.append({"config": asdict(config), "training": record,
                         "probabilities": probabilities.tolist(), "meets_requirement": success,
                         "reason": "默认配置" if len(attempts) == 0 else "首次未达标，增加步数并从相同初值重跑"})
        if success:
            break
    checks.record("synthetic/mlp_targets_above_0.9", success,
                  target_probabilities=probabilities[torch.arange(len(data.y)), data.y].tolist())
    check_first_step(checks, "synthetic/mlp", record)

    baseline = TrainableBigram(5)
    # 比较完全相同的两个预测目标，仅把输入信息压缩为最后一个token。
    last_token = data.x[:, -1]
    baseline_record = train(baseline, last_token, data.y, steps=1000)
    check_first_step(checks, "synthetic/bigram", baseline_record)
    with torch.no_grad():
        baseline_probabilities = baseline(last_token).softmax(-1)
    checks.close("synthetic/bigram_identical_context_distributions",
                 baseline_probabilities[0], baseline_probabilities[1], atol=0, rtol=0)
    checks.record("synthetic/same_prediction_targets", data.y.tolist() == [3, 4]
                  and last_token.tolist() == [2, 2])
    checks.record("synthetic/mlp_lower_loss",
                  record["history"][-1]["train"]["loss"] < baseline_record["history"][-1]["train"]["loss"])
    return {"purpose": "已见两条样本的拟合与表达能力对照，不是验证集泛化实验",
            "vocab": tokenizer.vocab, "documents": raw, "metadata": metadata,
            "data": data.to_dict(), "mlp_attempts": attempts,
            "bigram": {"input_ids": last_token.tolist(), "target_ids": data.y.tolist(),
                       "training": baseline_record, "probabilities": baseline_probabilities.tolist()},
            "explanation": "bigram只看到同一个末字爱，必然查同一参数行；MLP还可使用我/你。"}


def execute(args) -> dict:
    torch.set_num_threads(1)
    raw = json.loads(args.documents.read_text(encoding="utf-8"))
    tokenizer = CharTokenizer.load(args.vocab)
    config = ModelConfig(len(tokenizer.vocab), args.context_length, args.embedding_dim,
                         args.hidden_dim, args.seed)
    groups, metadata = prepare_data(raw, tokenizer, config.context_length)
    model = ContextMLP(config)
    train_data, val_data = groups["train"], groups["val"]
    training = train(model, train_data.x, train_data.y, learning_rate=args.learning_rate,
                     steps=args.steps, log_every=args.log_every, validation=(val_data.x, val_data.y))
    checks = run_checks()
    check_first_step(checks, "main", training)

    # 单个loss口径不依赖评估分批方式，最后一个小batch也正确加权。
    for batch_size in (1, 3):
        split_result = evaluate(model, train_data.x, train_data.y, batch_size=batch_size)
        checks.close(f"main/eval_chunk_{batch_size}", split_result["loss"],
                     training["history"][-1]["train"]["loss"])
    synthetic = synthetic_experiment(checks)
    generations = []
    for i, method in enumerate(("argmax", "sample")):
        context = train_data.samples[i % len(train_data.samples)]["context_text"]
        generated = generate(model, tokenizer, context, args.num_new_tokens, method=method, seed=123)
        replay = generate(model, tokenizer, context, args.num_new_tokens, method=method, seed=123)
        checks.record(f"generation/replay/{method}", generated == replay)
        tokens = tokenizer.encode(context)
        for step, entry in enumerate(generated["trace"]):
            checks.record(f"generation/context/{method}/{step}",
                          entry["context_ids"] == tokens[-config.context_length:])
            checks.close(f"generation/probability_sum/{method}/{step}", sum(entry["probabilities"]), 1.)
            tokens.append(entry["next_id"])
        checks.record(f"generation/decoded/{method}", tokenizer.decode(tokens) == generated["text"])
        generations.append(generated)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "model.pt"
    save_checkpoint(checkpoint_path, model, tokenizer)
    loaded, loaded_tokenizer = load_checkpoint(checkpoint_path)
    with torch.no_grad():
        original_logits = model(train_data.x)
        loaded_logits = loaded(train_data.x)
    checks.close("checkpoint/logits", loaded_logits, original_logits, atol=0, rtol=0)
    checks.record("checkpoint/vocab_config", loaded_tokenizer.vocab == tokenizer.vocab
                  and loaded.config == model.config)
    checks.record("checkpoint/new_parameter_objects", all(a is not b for a, b in
                  zip(model.parameters(), loaded.parameters())))

    sources = {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
               for name, path in (("documents", args.documents), ("vocab", args.vocab))}
    result = {"format_version": 1, "environment": {"python": platform.python_version(),
              "torch": str(torch.__version__), "device": "cpu", "dtype": "float64", "threads": 1},
              "config": asdict(config), "initialization": {"embedding": "normal(0,0.1)",
              "linear_weights": "PyTorch nn.Linear default", "biases": "zeros", "seed": config.seed},
              "sources": sources, "documents": raw, "vocab": tokenizer.vocab, "metadata": metadata,
              "data": {key: value.to_dict() for key, value in groups.items()},
              "parameter_info": model.parameter_info(), "training": training,
              "final_parameters": snapshot(model), "synthetic": synthetic, "generations": generations,
              "checkpoint": {"file": checkpoint_path.name, "config": asdict(loaded.config),
              "vocab": loaded_tokenizer.vocab, "original_logits": original_logits.tolist(),
              "loaded_logits": loaded_logits.tolist()},
              "scope": "授权实现与自检；尚未完成独立教学review、代码理解复盘或G1"}
    write_json(args.output_dir / "results.json", result)
    checks.record("serialization/results_roundtrip", json.loads(
        (args.output_dir / "results.json").read_text(encoding="utf-8")) == result)
    report = checks.result()
    write_json(args.output_dir / "checks.json", report)
    if json.loads((args.output_dir / "checks.json").read_text(encoding="utf-8")) != report:
        raise RuntimeError("checks.json 读回不一致")
    if not report["all_passed"]:
        failed = [item["name"] for item in report["checks"] if not item["passed"]]
        raise RuntimeError(f"检查失败：{failed}；详细结果见 {args.output_dir / 'checks.json'}")
    return {"checks": report["count"], "passed": report["passed"],
            "train_loss": training["history"][-1]["train"]["loss"],
            "val_loss": training["history"][-1]["val"]["loss"],
            "synthetic_mlp_loss": synthetic["mlp_attempts"][-1]["training"]["history"][-1]["train"]["loss"],
            "synthetic_bigram_loss": synthetic["bigram"]["training"]["history"][-1]["train"]["loss"],
            "output_dir": str(args.output_dir.resolve())}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--documents", type=Path, default=DEFAULT_DATA / "documents.json")
    result.add_argument("--vocab", type=Path, default=DEFAULT_DATA / "vocab.json")
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--context-length", type=int, default=2)
    result.add_argument("--embedding-dim", type=int, default=4)
    result.add_argument("--hidden-dim", type=int, default=16)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--learning-rate", type=float, default=.1)
    result.add_argument("--steps", type=int, default=1000)
    result.add_argument("--log-every", type=int, default=50)
    result.add_argument("--num-new-tokens", type=int, default=12)
    return result


def main():
    summary = execute(parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
