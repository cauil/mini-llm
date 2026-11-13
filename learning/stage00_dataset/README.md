# 阶段 0C：训练样本构造器

依据[任务书 v1](../../tasks/阶段0C_训练样本构造任务书.md)实现。用户于 2026-11-11 明确请求编码 agent 实现；实现和自动检查不代替教学方后续 review，也不代表用户理解验收通过。未替用户回答任务书的理解题。

Python 3.9 或以上，仅使用标准库。直接导入复用 0B 的 [CharTokenizer](../stage00_char_codec/tokenizer.py)，不复制编码逻辑、不修改词表。

## 运行

从项目根目录运行（使用 `-m` 保持跨目录导入一致，不支持直接运行文件路径）：

```bash
cd /Users/lcf/llm
python3 -B -m learning.stage00_dataset.prepare
python3 -B -m learning.stage00_dataset.check_dataset
python3 -B learning/stage00_char_codec/check_codec.py
```

默认读取本目录 [documents.json](./documents.json)、[vocab.json](./vocab.json)，T=3。生成本目录 `output/` 中的 `train_samples.json`、`val_samples.json` 和 `metadata.json`，每次覆盖而非追加。公开示例应产生训练 3 条、验证 1 条。

也可以指定输入及输出目录：

```bash
python3 -B -m learning.stage00_dataset.prepare \
  --documents learning/stage00_dataset/documents.json \
  --vocab learning/stage00_dataset/vocab.json \
  --context-length 3 \
  --output-dir learning/stage00_dataset/output
```

## 接口与设计

实现位于 [prepare.py](./prepare.py)：

- `build_windows(tokens, T)`：返回包含 `start/x/y` 的列表；只负责单文档窗口，不读写文件。起点为 `range(N-T)`，每条输入、目标分别取 `[start:start+T]` 和 `[start+1:start+T+1]`。N≤T 时没有窗口。
- `prepare_dataset(data, tokenizer, T)`：先校验完整文档归属，再对每篇文档完整编码，随后调用窗口函数。返回 `train`、`val` 样本列表和 `metadata`。每条样本加上 `doc_id`。
- `prepare_files(documents_path, vocab_path, T, output_dir)`：读取 JSON、加载固定 tokenizer、生成内存结果、保存三个 JSON，并返回同一内存结果。

输入顶层仅包含 `train`、`val` 两个文档列表；id 为全局唯一的非空字符串，text 为字符串。文档可携带其他字段，但输出只使用 id/text。空组、空文档允许。保留组内文档顺序和每篇的起点顺序，不排序、不拼接、不补齐、不加入特殊 token、不随机划分。

metadata 记录 `T`、固定 `stride=1`、实际 `vocab`、`train_doc_ids`、`val_doc_ids`，以及 `documents` 列表；每项包含 `doc_id/split/token_count/sample_count`，包括零样本文档。它用于结合输入原文和样本起点追踪来源，不代替原文备份。

参数、结构、重复 id 错误抛出 `ValueError`。未知字符抛出 `KeyError`，包含分组、文档 id 和未知字符；即使文档太短也先检查全部字符。文件、JSON、词表错误直接向调用方报告。非法输入不会开始写入结果；三个输出不是事务写入，磁盘写入失败会报错，不能保证三文件原子更新。

本模块仅按 id 校验分组，不做内容去重。train/val 可共享词语；不同 id 的相同内容也不会被当成 id 冲突。实际大规模语料需要另行设计内容去重与流水线。

## 工程断言

[check_dataset.py](./check_dataset.py) 使用标准库 unittest，包含 14 个测试方法及参数化子检查：

- 两个不同起点的手算完整 x/y；公开例子所有样本及最后合法起点。
- N=0、T−1、T、T+1 和更长文本；T=1；空组与空文档。
- 重复字符、空格和换行；逐样本解码追溯原文，核对来源、顺序及长度。
- 同组/跨组短文档不拼接；同组/跨组重复 id；非法 T、文档结构和 token 列表。
- 训练/验证中的未知字符（包括太短的未知字符文档）；固定词表不变。
- 相同词语/内容但不同 id 的合法输入；训练和验证归属隔离。
- 保存再读与内存相同；重复运行结果和文件字节一致；改为空输入会覆盖旧样本；metadata 含零样本文档且数量吻合。
- 验证文档编码失败时不覆盖已有结果。

所有测试写入使用临时目录，结束后自动清理。工程结果与实际运行记录见根目录[学习验收记录](../../学习验收记录.md)；用户手算、变体解释和 review 状态另记。

## 后续接入与参考

此接口提供后续字符 bigram/GPT 的单样本列表，不引入 batch、PyTorch 或训练循环。对应 Karpathy《[Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)》的 block_size、x/y 切片，以及 [CS336 Assignment 1 Basics](https://github.com/stanford-cs336/assignment1-basics) 的数据加载主题。实现依据本地任务书与现有 tokenizer，未复制课程代码。它是最终预训练数据准备的教学模块，不等于完成 BPE 或大规模语料流水线。
