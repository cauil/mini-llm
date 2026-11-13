"""运行阶段 1：读文档、拟合计数模型、评估、生成并保存结果。"""

import argparse
import json
from pathlib import Path

from learning.stage00_char_codec.tokenizer import CharTokenizer
from learning.stage01_count_bigram.bigram import CountBigram, validate_groups


def build_run(data: object, tokenizer: CharTokenizer, prompt: str,
              num_new_tokens: int = 12, seed: int = 42) -> dict:
    groups = validate_groups(data)
    model = CountBigram(tokenizer).fit(groups["train"])
    return {
        "model": model.to_dict(),
        "evaluation": {split: model.evaluate(groups[split]) for split in ("train", "val")},
        "generations": [model.generate(prompt, num_new_tokens, method, seed)
                        for method in ("argmax", "sample")],
        "conventions": {
            "context_tokens": 1, "cross_document_pairs": False,
            "boundary_tokens": False, "log_base": "e",
            "mean_over": "all_prediction_positions",
            "stop_after": "num_new_tokens",
        },
    }


def run_files(documents_path: Path, vocab_path: Path, output_dir: Path,
              prompt: str, num_new_tokens: int = 12, seed: int = 42) -> dict:
    with documents_path.open(encoding="utf-8") as file:
        data = json.load(file)
    tokenizer = CharTokenizer.load(vocab_path)
    result = build_run(data, tokenizer, prompt, num_new_tokens, seed)
    # 全部输入与计算成功后才保存；无效输入不会覆盖已有结果。
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(serialized, encoding="utf-8")
    return result


def main() -> None:
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=folder / "documents.json")
    parser.add_argument("--vocab", type=Path, default=folder / "vocab.json")
    parser.add_argument("--output-dir", type=Path, default=folder / "output")
    parser.add_argument("--prompt", default="春")
    parser.add_argument("--num-new-tokens", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = run_files(args.documents, args.vocab, args.output_dir,
                           args.prompt, args.num_new_tokens, args.seed)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    for split, metrics in result["evaluation"].items():
        print(f"{split}: 预测位置={metrics['pair_count']}，平均 NLL={metrics['mean_nll']}")
    for generated in result["generations"]:
        print(f"{generated['method']}: {generated['text']}")
    print(f"结果：{(args.output_dir / 'results.json').resolve()}")


if __name__ == "__main__":
    main()
