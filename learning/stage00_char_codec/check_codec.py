"""工程验收：用手工结果和往返性质核对模块，不代表用户理解已验收。"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from tokenizer import CharTokenizer, decode, encode, load_vocab, save_vocab


def require(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
    print("通过：", label)


def require_raises(error_type, action, label: str, value=None) -> None:
    try:
        action()
    except error_type as error:
        if value is not None and repr(value) not in str(error):
            raise AssertionError(f"异常未指出问题值：{value!r}") from error
        print("通过：", label)
    else:
        raise AssertionError(label)


def main() -> None:
    vocab = load_vocab(Path(__file__).with_name("vocab.json"))
    require(encode("鱼吃肉", vocab) == [2, 1, 3], "编码与手算一致")
    require(decode([3, 1, 2], vocab) == "肉吃鱼", "解码与手算一致")
    require(encode("猫猫吃鱼", vocab) == [0, 0, 1, 2], "重复字符使用相同编号")

    examples = ["", "猫", "猫吃鱼", "鱼吃肉", "猫猫吃鱼"]
    require(
        all(decode(encode(text, vocab), vocab) == text for text in examples),
        "词表覆盖的文本可以往返还原，包括空文本",
    )

    with TemporaryDirectory() as folder:
        saved_path = Path(folder) / "saved_vocab.json"
        save_vocab(vocab, saved_path)
        restored = load_vocab(saved_path)
        require(
            restored == vocab and encode("猫吃肉", restored) == [0, 1, 3],
            "词表保存/加载后映射与编码保持一致",
        )

    changed_vocab = {**vocab, "猫": 3, "肉": 0}
    require(
        decode([0, 1, 2], changed_vocab) == "肉吃鱼", "更换词表会改变既有编号的含义"
    )

    try:
        decode([0], {"猫": 0, "肉": 0})
    except ValueError:
        print("通过：拒绝无法唯一解码的重复编号词表")
    else:
        raise AssertionError("应拒绝重复编号词表")

    tokenizer = CharTokenizer(vocab)
    require(tokenizer.encode("肉鱼猫") == [3, 2, 0], "新例子具体编号正确")
    require(tokenizer.encode("") == [] and tokenizer.decode([]) == "", "空输入")
    rich = CharTokenizer({**vocab, " ": 4, "\n": 5})
    require(
        rich.encode("猫猫\n 鱼") == [0, 0, 5, 4, 2]
        and rich.decode([0, 0, 5, 4, 2]) == "猫猫\n 鱼",
        "重复字符、空格和换行原样保留",
    )
    require_raises(KeyError, lambda: tokenizer.encode("狗"), "拒绝未知字符", "狗")
    for token_id in [-1, len(vocab)]:
        require_raises(KeyError, lambda: tokenizer.decode([token_id]),
                       f"拒绝无效编号 {token_id}", token_id)
    for token_id in [0.0, True, "0"]:
        require_raises(ValueError, lambda: tokenizer.decode([token_id]),
                       f"拒绝非整数编号 {token_id!r}", token_id)

    original = dict(vocab)
    fixed = CharTokenizer(original)
    original["猫"], original["肉"] = 3, 0
    exposed = fixed.vocab
    exposed["鱼"] = 99
    fixed.encode("猫吃鱼")
    fixed.encode("鱼吃肉")
    require(
        fixed.vocab == vocab and fixed.encode("猫吃鱼") == [0, 1, 2]
        and fixed.decode([0, 1, 2]) == "猫吃鱼",
        "原字典及导出副本的修改、多次编码均不改变固定映射",
    )

    with TemporaryDirectory() as folder:
        path = Path(folder) / "vocab.json"
        fixed.save(path)
        restored = CharTokenizer.load(path)
        require(
            restored.vocab == vocab and restored.encode("鱼吃肉") == [2, 1, 3]
            and restored.decode([3, 1, 2]) == "肉吃鱼",
            "外部字典改动后保存加载仍保持原始映射",
        )
        invalid_vocabs = [
            {}, {"猫": 0, "肉": 0}, {"猫": 0, "鱼": 2}, {"猫": -1},
            {"猫咪": 0}, {"": 0}, {"猫": 0.5}, {"猫": "0"},
            {"猫": False}, [],
        ]
        for bad in invalid_vocabs:
            require_raises(ValueError, lambda: CharTokenizer(bad),
                           f"创建时拒绝非法词表 {bad!r}")
            path.write_text(json.dumps(bad), encoding="utf-8")
            require_raises(ValueError, lambda: CharTokenizer.load(path),
                           f"加载时拒绝非法词表 {bad!r}")
        require_raises(ValueError, lambda: encode("猫", {"猫": 3}),
                       "旧编码接口同样校验词表")
        require_raises(ValueError, lambda: save_vocab({"猫": 3}, path),
                       "旧保存接口同样校验词表")
        require_raises(ValueError, lambda: load_vocab(path),
                       "旧加载接口同样校验词表")

    print("全部检查通过；理解验收需另行完成。")


if __name__ == "__main__":
    main()
