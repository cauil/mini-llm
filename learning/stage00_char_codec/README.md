# 阶段 0B：固定词表的字符编码器

当前实现为 `tokenizer.py`。用户于 2026-11-11 提交首版，随后明确请求教学方优化；本次由教学 agent 补齐词表校验、固定映射封装、检查和说明。既有 demo/检查来自先前教学参考并经适配，不将其称为用户独立编写。工程检查通过，用户理解验收仍待完成。

## 用法

Python 3.9 或以上，仅使用标准库。在本目录的 Python 脚本中：

```python
from tokenizer import CharTokenizer

vocab = {"猫": 0, "吃": 1, "鱼": 2, "肉": 3}
tokenizer = CharTokenizer(vocab)
ids = tokenizer.encode("鱼吃肉")  # [2, 1, 3]
text = tokenizer.decode(ids)      # "鱼吃肉"
```

`CharTokenizer` 建立时复制词表，反向映射也只构造一次。修改调用方的原始字典，或者修改 `tokenizer.vocab` 返回的副本，不会影响此实例。后续数据使用同一个实例；`tokenizer.save(path)` 保存，`CharTokenizer.load(path)` 加载。

旧接口 `encode(text, vocab)`、`decode(ids, vocab)`、`save_vocab(vocab, path)`、`load_vocab(path)` 继续可用且包含校验。它们每次根据传入词表独立工作，不保存跨调用状态；更换传入词表本来就可能改变结果。需要固定映射时使用上面的实例接口。

## 范围与异常

- 每个 Python 字符为一个 token。词表必须为非空字典，key 长度为 1，编号为唯一整数且连续覆盖 0 到 V−1。
- 非法词表在创建或加载时抛出 `ValueError`；浮点数和布尔值不能作为编号。
- 未知字符、负编号和越界编号抛出 `KeyError`，包含问题值；非整数解码编号抛出 `ValueError`。
- 顺序、重复字符、空格、换行保留；空字符串和空编号列表可以互相转换。
- JSON 使用 UTF-8，保存原映射。文件不存在、无效 JSON 等保留标准库异常。
- 不含自动建词表、UNK、BPE、训练窗口或神经网络。

## 运行与验收

```bash
python3 -B /Users/lcf/llm/learning/stage00_char_codec/demo.py
python3 -B /Users/lcf/llm/learning/stage00_char_codec/check_codec.py
```

2026-11-11 修订版：41 项检查通过，demo 正常运行。保留原有 7 项检查，补充新例子、空白、非法编号、创建/加载非法词表、外部字典修改隔离及旧接口校验。所有文件写入检查均使用临时目录。通过工程检查不等于理解验收通过。

建议阅读顺序：`vocab.json` → `tokenizer.py` 的 encode/decode 方法 → 构造函数和 vocab 属性 → 校验及保存加载 → `check_codec.py`。

## 课程与最终作业关联

Karpathy《Let's build GPT》的字符词表、stoi/itos、encode/decode；CS336 Assignment 1 的 tokenizer 主题：

- https://www.youtube.com/watch?v=kCc8FmEb1nY
- https://github.com/stanford-cs336/assignment1-basics

此模块用于后续字符级样本和 bigram。最终作业还需实现 BPE 与参考 tokenizer 编码流程，本模块不替代它们。下一步先完成理解验收，再进入 0C。
