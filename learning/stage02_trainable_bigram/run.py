"""运行公开数据训练、计数基线与全覆盖频率实验，保存真实结果。"""

import argparse
import json
import math
from pathlib import Path
import platform

import torch

from learning.stage00_char_codec.tokenizer import CharTokenizer
from learning.stage01_count_bigram.bigram import CountBigram
from .data import prepare_data
from .model import TrainableBigram, generate
from .training import evaluate, train


STAGE_DIR = Path(__file__).resolve().parent
BASELINE_DIR = STAGE_DIR.parent / "stage01_count_bigram"
OUTPUT_DIR = STAGE_DIR / "output"


def summary(evaluation: dict) -> dict:
    return {key: evaluation[key]
            for key in ("pair_count", "nll_sum", "mean_nll", "reason")}


def frequency_experiment(steps: int = 2000) -> dict:
    tokenizer = CharTokenizer({"甲": 0, "乙": 1})
    documents = []
    for pair, repetitions in (("甲甲", 3), ("甲乙", 1), ("乙甲", 1), ("乙乙", 3)):
        for repeat in range(repetitions):
            documents.append({"id": f"{pair}-{repeat}", "text": pair})
    data = prepare_data(tokenizer, {"train": documents, "val": []})
    counts = [[0, 0], [0, 0]]
    for sample in data["train"].samples:
        counts[sample["current_id"]][sample["target_id"]] += 1
    frequencies = [[count / sum(row) for count in row] for row in counts]
    # 频率只用于比较，不写入参数；模型仍从零开始反向传播学习。
    model = TrainableBigram(2)
    trained = train(model, data["train"], data["val"], steps=steps,
                    learning_rate=0.3, log_every=max(1, steps))
    probabilities = [model.distribution(i) for i in range(2)]
    error = max(abs(probabilities[i][j] - frequencies[i][j])
                for i in range(2) for j in range(2))
    return {"vocab": tokenizer.vocab, "documents": documents,
            "counts": counts, "target_unsmoothed_frequencies": frequencies,
            "config": trained["config"], "history": trained["history"],
            "final_logits": model.snapshot(), "final_probabilities": probabilities,
            "max_absolute_error": error, "tolerance": 0.02, "passed": error <= 0.02}


def run_experiment(vocab_path: Path = BASELINE_DIR / "vocab.json",
                   documents_path: Path = BASELINE_DIR / "documents.json",
                   learning_rate: float = 0.3, steps: int = 300,
                   log_every: int = 10, dtype: torch.dtype = torch.float64,
                   seed: int = 42, prompt: str = None,
                   num_new_tokens: int = 20, verbose: bool = False) -> dict:
    tokenizer = CharTokenizer.load(vocab_path)
    raw = json.loads(Path(documents_path).read_text(encoding="utf-8"))
    data = prepare_data(tokenizer, raw)
    model = TrainableBigram(len(tokenizer.vocab), dtype=dtype)
    if verbose:
        print("初始化参数：" + json.dumps(model.parameter_info(), ensure_ascii=False))
    trained = train(model, data["train"], data["val"],
                    learning_rate=learning_rate, steps=steps, log_every=log_every)
    final = {split: evaluate(model, dataset) for split, dataset in data.items()}
    # 只读复用阶段 1 的实现；不加载其他数据对应的旧结果，不改写旧输出。
    baseline = CountBigram(tokenizer).fit(raw["train"])
    baseline_metrics = {split: summary(baseline.evaluate(raw[split]))
                        for split in ("train", "val")}
    if prompt is None:
        prompt = tokenizer.decode([0])
    generations = [generate(model, tokenizer, prompt, num_new_tokens, method, seed)
                   for method in ("argmax", "sample")]
    result = {
        "schema_version": 1,
        "environment": {"python": platform.python_version(), "pytorch": str(torch.__version__),
                        "device": "cpu", "dtype": str(dtype),
                        "num_threads": torch.get_num_threads()},
        "sources": {"vocab": str(Path(vocab_path).resolve()),
                    "documents": str(Path(documents_path).resolve())},
        "vocab": tokenizer.vocab,
        "data": {split: dataset.to_dict() for split, dataset in data.items()},
        "training": trained,
        "model": {"logits": model.snapshot(),
                  "probabilities": [model.distribution(i) for i in range(model.vocab_size)]},
        "final_evaluation": final,
        "comparison": {"count_add_one": baseline_metrics,
                       "initial_trainable": {s: trained["history"][0][s] for s in ("train", "val")},
                       "final_trainable": {s: summary(final[s]) for s in ("train", "val")}},
        "generations": generations,
        "frequency_experiment": frequency_experiment(),
    }
    return result


def save_results(result: dict, path: Path) -> None:
    path = Path(path).resolve()
    if not path.is_relative_to(STAGE_DIR) or path.suffix != ".json":
        raise ValueError("结果必须保存为本阶段目录内的 JSON，不能覆盖旧阶段产物")
    # 先序列化，拒绝 NaN/Inf；失败时不截断现有文件。
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if json.loads(path.read_text(encoding="utf-8")) != result:
        raise RuntimeError("结果保存后的重读核对失败")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=Path, default=BASELINE_DIR / "vocab.json")
    parser.add_argument("--documents", type=Path, default=BASELINE_DIR / "documents.json")
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt")
    parser.add_argument("--num-new-tokens", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(STAGE_DIR):
        parser.error("--output-dir 必须位于本阶段目录内")
    # 极小矩阵无需多个 CPU 线程，减少并行调度开销；仅运行入口设置。
    torch.set_num_threads(1)
    try:
        result = run_experiment(
            args.vocab, args.documents, args.learning_rate, args.steps, args.log_every,
            getattr(torch, args.dtype), args.seed, args.prompt, args.num_new_tokens, verbose=True,
        )
    except (ValueError, FloatingPointError, RuntimeError) as error:
        save_results({"status": "failed", "error": str(error),
                      "config": {k: str(v) if isinstance(v, Path) or
                                 (isinstance(v, float) and not math.isfinite(v)) else v
                                 for k, v in vars(args).items()}},
                     args.output_dir / "error.json")
        parser.exit(1, f"运行失败：{error}\n")
    save_results(result, args.output_dir / "results.json")
    print(json.dumps({"comparison": result["comparison"],
                      "frequency_max_error": result["frequency_experiment"]["max_absolute_error"],
                      "generations": result["generations"],
                      "saved": str((args.output_dir / "results.json").resolve())},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
