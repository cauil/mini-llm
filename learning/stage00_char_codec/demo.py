"""展示课堂例子；从任何工作目录运行都读取脚本旁边的词表。"""

from pathlib import Path

from tokenizer import decode, encode, load_vocab


def main() -> None:
    vocab = load_vocab(Path(__file__).with_name("vocab.json"))
    print("词表：", vocab)
    for text in ["猫吃鱼", "猫吃肉", "鱼吃肉", "猫猫吃鱼"]:
        token_ids = encode(text, vocab)
        print(f"{text!r} → {token_ids} → {decode(token_ids, vocab)!r}")

    print("[3, 1, 2] 解码为：", decode([3, 1, 2], vocab))

    # 只改一个副本，不覆盖保存的原词表。
    changed_vocab = {**vocab, "猫": 3, "肉": 0}
    print("同一份 [0, 1, 2]：")
    print("  原词表解码：", decode([0, 1, 2], vocab))
    print("  新词表解码：", decode([0, 1, 2], changed_vocab))


if __name__ == "__main__":
    main()
