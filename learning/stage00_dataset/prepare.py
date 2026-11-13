"""固定词表下的文档级训练/验证样本准备。"""

import argparse
import json
from pathlib import Path
from typing import TypedDict

from learning.stage00_char_codec.tokenizer import CharTokenizer


class Window(TypedDict):
    start: int
    x: list[int]
    y: list[int]


class Sample(Window):
    doc_id: str


class Document(TypedDict):
    id: str
    text: str


class DocumentStats(TypedDict):
    doc_id: str
    split: str
    token_count: int
    sample_count: int


class Metadata(TypedDict):
    T: int
    stride: int
    vocab: dict[str, int]
    train_doc_ids: list[str]
    val_doc_ids: list[str]
    documents: list[DocumentStats]


class PreparedDataset(TypedDict):
    train: list[Sample]
    val: list[Sample]
    metadata: Metadata


def _validate_context_length(T: int) -> None:
    if type(T) is not int or T <= 0:
        raise ValueError(f"T 必须是正整数且不能为布尔值，收到：{T!r}")


def build_windows(tokens: list[int], T: int) -> list[Window]:
    """单篇文档内枚举所有完整窗口，start 是文档内的 token 下标。"""
    _validate_context_length(T)
    if not isinstance(tokens, list) or any(type(token) is not int or token < 0
                                           for token in tokens):
        raise ValueError("tokens 必须是非负整数列表，不能包含布尔值")
    return [
        {
            "start": start,
            "x": tokens[start:start + T],
            "y": tokens[start + 1:start + T + 1],
        }
        for start in range(len(tokens) - T)
    ]


def _validate_documents(data: object) -> dict[str, list[Document]]:
    if not isinstance(data, dict) or set(data) != {"train", "val"}:
        raise ValueError("数据必须是只含 train 和 val 两组的 JSON 对象")
    groups: dict[str, list[Document]] = {}
    seen: set[str] = set()
    for split in ("train", "val"):
        raw_documents = data[split]
        if not isinstance(raw_documents, list):
            raise ValueError(f"{split} 必须是文档列表")
        documents: list[Document] = []
        for index, raw in enumerate(raw_documents):
            location = f"{split}[{index}]"
            if not isinstance(raw, dict):
                raise ValueError(f"{location} 必须是包含 id 和 text 的对象")
            doc_id = raw.get("id")
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError(f"{location} 的 id 必须为非空字符串，收到：{doc_id!r}")
            if doc_id in seen:
                raise ValueError(f"文档 id 重复（含跨组重复）：{doc_id!r}")
            text = raw.get("text")
            if not isinstance(text, str):
                raise ValueError(f"文档 {doc_id!r} 的 text 必须为字符串")
            seen.add(doc_id)
            documents.append({"id": doc_id, "text": text})
        groups[split] = documents
    return groups


def prepare_dataset(data: object, tokenizer: CharTokenizer, T: int) -> PreparedDataset:
    """校验已划分文档，使用同一个 tokenizer，返回样本及来源统计。"""
    _validate_context_length(T)
    groups = _validate_documents(data)
    samples: dict[str, list[Sample]] = {"train": [], "val": []}
    stats: list[DocumentStats] = []
    for split in ("train", "val"):
        for document in groups[split]:
            doc_id = document["id"]
            try:
                tokens = tokenizer.encode(document["text"])
            except KeyError as error:
                raise KeyError(
                    f"{split} 文档 {doc_id!r} 含未知字符：{error.args[0]!r}"
                ) from error
            windows = build_windows(tokens, T)
            for window in windows:
                samples[split].append({
                    "doc_id": doc_id,
                    "start": window["start"],
                    "x": window["x"],
                    "y": window["y"],
                })
            stats.append({
                "doc_id": doc_id,
                "split": split,
                "token_count": len(tokens),
                "sample_count": len(windows),
            })
    return {
        "train": samples["train"],
        "val": samples["val"],
        "metadata": {
            "T": T,
            "stride": 1,
            "vocab": tokenizer.vocab,
            "train_doc_ids": [doc["id"] for doc in groups["train"]],
            "val_doc_ids": [doc["id"] for doc in groups["val"]],
            "documents": stats,
        },
    }


def prepare_files(
    documents_path: Path, vocab_path: Path, T: int, output_dir: Path
) -> PreparedDataset:
    """先完成全部校验和编码，再覆盖写入三个 JSON 文件。"""
    _validate_context_length(T)
    with documents_path.open(encoding="utf-8") as file:
        data = json.load(file)
    tokenizer = CharTokenizer.load(vocab_path)
    prepared = prepare_dataset(data, tokenizer, T)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        ("train_samples.json", prepared["train"]),
        ("val_samples.json", prepared["val"]),
        ("metadata.json", prepared["metadata"]),
    )
    for filename, content in outputs:
        with (output_dir / filename).open("w", encoding="utf-8") as file:
            json.dump(content, file, ensure_ascii=False, indent=2)
            file.write("\n")
    return prepared


def main() -> None:
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=folder / "documents.json")
    parser.add_argument("--vocab", type=Path, default=folder / "vocab.json")
    parser.add_argument("--context-length", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=folder / "output")
    args = parser.parse_args()
    result = prepare_files(
        args.documents, args.vocab, args.context_length, args.output_dir
    )
    print(f"已保存：train={len(result['train'])}，val={len(result['val'])}")
    print(f"输出目录：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
