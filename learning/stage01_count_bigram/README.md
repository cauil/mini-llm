# 阶段 1：字符级计数 bigram

按 `tasks/阶段1_计数bigram任务书.md` v1 实现。使用 Python 3.9+ 标准库，复用阶段 0 的 CharTokenizer，不需要安装依赖或使用 GPU。

## 运行

在 `/Users/lcf/llm` 执行：

```bash
python3 -B -m learning.stage01_count_bigram.check_bigram
python3 -B -m learning.stage01_count_bigram.run
```

自定义输入、提示词和输出位置：

```bash
python3 -B -m learning.stage01_count_bigram.run \
  --documents learning/stage01_count_bigram/documents.json \
  --vocab learning/stage01_count_bigram/vocab.json \
  --prompt 春风 --num-new-tokens 20 --seed 42 \
  --output-dir /tmp/stage01-bigram-demo
```

默认用同一个提示词分别展示 argmax 和 sample。两种生成均保留提示词，新增指定数量的字符后停止；没有起止符。

## 文件与接口

| 文件 | 用途 |
|---|---|
| `bigram.py` | CountBigram：计数、平滑、查询、生成与评估；文档校验 |
| `run.py` | 读取输入，串起完整实验，保存结果；CLI 入口 |
| `check_bigram.py` | 18 项自动检查及子检查 |
| `documents.json` / `vocab.json` | 课堂三字符公开数据和固定词表 |
| `output/results.json` | 实际运行保存的模型、指标、逐位置明细和生成配置/结果 |

主要接口：

- `CountBigram(tokenizer).fit(train_documents)`：从头拟合；只接收训练文档，返回模型自身。
- `distribution(current_id)`：返回下一 token 分布副本。
- `sample_next(current_id, rng)`：从一行概率抽样；调用方持续使用同一个 `random.Random`，可用于固定行频率检查。
- `generate(prompt, num_new_tokens, method="sample", seed=0)`：返回配置、完整文本及 token 列表。argmax 并列时选编号最小者。
- `evaluate(documents)`：返回总预测位置数、NLL 总和、平均 NLL，以及每篇文档/位置的明细。
- `to_dict()`：返回词表、计数、概率和规则的副本。
- `build_run(...)` / `run_files(...)`：完整实验的内存入口和文件入口。默认生成种子为 42。

示例：

```python
from learning.stage00_char_codec.tokenizer import CharTokenizer
from learning.stage01_count_bigram.bigram import CountBigram

tokenizer = CharTokenizer({"春": 0, "风": 1, "雨": 2})
model = CountBigram(tokenizer).fit([{"id": "train", "text": "春风春"}])
print(model.distribution(0))
print(model.generate("春", 2, method="argmax"))
print(model.evaluate([{"id": "val", "text": "春雨风"}]))
```

## 与课堂概念的对应

`fit` 为每篇文档分别编码并遍历相邻对。矩阵的行是当前 token，列是下一个 token；原始计数保留不变，另建加一平滑概率表。重新拟合会替换旧计数；输入校验失败保留旧模型。

`run.py` 只把 `groups["train"]` 交给 `fit`。train/val 都可以交给 `evaluate`，但评估没有修改计数的操作；读取的是固定概率表中的真实目标项。评估也不调用采样。

生成每步读取 `tokens[-1]`，抽到的新 token 追加后成为下一步查询的行。采样生成器只在循环外初始化一次；argmax 并列固定取最小编号。固定种子便于复现，不意味着不同种子一定给出不同结果。

多篇文档先累加 NLL 总和，再除以所有预测位置数。没有位置时，`mean_nll` 为 JSON `null` 并附带原因。文档首字符不单独评分；无起始符时，长度 N 的文档只贡献 max(0, N−1) 个位置。

统计直接读取完整文档，不使用 0C 的重叠窗口，以免重复计数。空白和换行也是普通字符，词表包含它们时原样处理。未知字符即使出现在单字符文档或提示词较早位置，也会报错。

此实现面向小规模学习数据，保存全部逐位置评估明细；未实现大语料流式处理、结束符、可调平滑强度或神经网络训练。

## 本次实际结果（2026-11-12）

执行环境：Python 3.9.6。公开训练文档为春风春、春风雨、雨风春，验证文档为春雨风。矩阵顺序为 [春, 风, 雨]。

```text
原始计数：
[[0, 2, 0],
 [2, 0, 1],
 [0, 1, 0]]

平滑概率：
[[1/5, 3/5, 1/5],
 [1/2, 1/6, 1/3],
 [1/4, 1/2, 1/4]]
```

| 分组 | 有效预测位置 | 平均 NLL |
|---|---:|---:|
| train | 6 | 0.6999508463133212 |
| val | 2 | 1.1512925464970227 |

验证位置 `春→雨` 和 `雨→风` 的概率分别为 0.2 和 0.5，与用户先前的手算一致。明细中 `target_position` 为文档内从 0 开始的目标下标，本例为 1、2。

提示词“春”，新增 12 个字符，seed=42，实际输出：

```text
argmax: 春风春风春风春风春风春风春
sample: 春风春风春风雨雨春风春风风
```

生成样例只用于观察行为，不用句子是否流畅判断工程正确性。

## 检查证据

2026-11-12 实际执行：

- `python3 -B -m learning.stage01_count_bigram.check_bigram`：18 个测试方法及子检查全部通过。
- 其中包括 100 组随机小语料与字符对 Counter 参考比较；固定“风”行抽样 10000 次，seed=20261112，各项频率误差不超过 0.03。
- 覆盖手算计数、非对称转移、未见字符行、空训练/验证、空文档、文档边界、空白与换行、未知字符、重复 id、只读副本、重复拟合、验证隔离、不等长文档平均、argmax 并列、连续采样随机状态、保存重读、重复运行以及实际 CLI。
- 回归 `python3 -B learning/stage00_char_codec/check_codec.py`：41 项通过。
- 回归 `python3 -B -m learning.stage00_dataset.check_dataset`：14 个测试方法及子检查通过。

测试中的文件写入使用临时目录。默认运行入口会覆盖本目录 `output/results.json`；无效输入在保存前报错，不覆盖已有输出。

## 参考与 AI 使用范围

- 需求依据：本项目阶段 1 任务书、课堂手算与本地《大模型作业.pdf》关联要求。
- 概念参考：[Karpathy makemore](https://github.com/karpathy/makemore)；[CS336 Assignment 1 Basics](https://github.com/stanford-cs336/assignment1-basics) 的语言建模目标作为后续参考。本次未复制这些仓库代码。
- 复用本项目现有 CharTokenizer；未修改阶段 0 实现。
- 用户明确请求“帮我完成阶段1 编码任务”后，由教学方 agent 完成本阶段代码、检查与文档，并实际运行验证。没有委派其他 agent。

当前状态为**已实现、工程 review 通过，本轮学习收束，共享计数矩阵独立复验待补**。2026-11-12 应用户请求完成审查：18 项测试方法及子检查复跑通过；另用 60 组不同词表大小/编号顺序的数据独立核对计数、概率、验证 NLL 与逐步 argmax，全部通过；已保存结果与重新计算一致。审查记录见 `reviews/阶段1_代码审查.md`。用户在计数代码讲解后确认理解，未记作全部独立验收通过。知识与 side chat 总结见 `/Users/lcf/llm/01-计数bigram.md`。
