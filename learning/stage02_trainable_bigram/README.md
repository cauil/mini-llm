# 阶段 2：可训练字符 bigram

按 `/Users/lcf/llm/tasks/阶段2_可训练bigram任务书.md` v1 实现。2026-11-13 用户明确授权教学方编码；当前为**工程 review 通过，本轮主体教学与代码复盘收束**，允许进入阶段 3；部分独立/延迟复验仍待后续完成。总结见 `/Users/lcf/llm/02-可训练bigram.md`。审查见 `/Users/lcf/llm/reviews/阶段2_代码审查.md`。

## 环境与运行

使用用户已有的 conda 环境 `pytorch`，本次实测为 Python 3.12.16、PyTorch 2.15.0。无需创建新的 PyTorch 环境，也未修改该环境的依赖。

从项目根目录执行：

```bash
cd /Users/lcf/llm
conda activate pytorch
python -B -m learning.stage02_trainable_bigram.run
python -B -m learning.stage02_trainable_bigram.check_bigram
```

如果终端未初始化 conda，可直接使用本机已验证的解释器：

```bash
/Users/lcf/.conda/envs/pytorch/bin/python -B -m learning.stage02_trainable_bigram.run
/Users/lcf/.conda/envs/pytorch/bin/python -B -m learning.stage02_trainable_bigram.check_bigram
```

默认：CPU、float64、全零参数、全量 batch、学习率 0.3、SGD（无动量/权重衰减）、300 次更新、每 10 步记录。默认提示词为词表编号 0 对应字符（公开数据为“春”），新增 20 个字符，采样种子 42。运行入口仅将 CPU 线程设为 1，减少极小矩阵的调度开销；模型/训练接口不改变全局线程设置。

默认输入直接复用：

- `/Users/lcf/llm/learning/stage01_count_bigram/vocab.json`
- `/Users/lcf/llm/learning/stage01_count_bigram/documents.json`

完整参数列表：

```bash
python -B -m learning.stage02_trainable_bigram.run --help
```

例如只查看初始模型，并将结果写入另一个子目录，保留默认训练结果：

```bash
python -B -m learning.stage02_trainable_bigram.run --steps 0 --output-dir learning/stage02_trainable_bigram/output/initial
```

每次运行还会执行一个独立的 2000 步全覆盖频率实验，因此 `--steps 0` 只控制主实验。允许 `--dtype float32`；默认 float64 方便与手算核对。输出必须位于本阶段目录内，重复运行覆盖同名结果，不覆盖旧阶段文件。

## 文件与阅读顺序

| 文件 | 职责 |
|---|---|
| `data.py` | 复用 CharTokenizer 和阶段 1 的文档校验；逐篇提取相邻对，保留来源 |
| `model.py` | 注册分数表、按输入编号查行、概率查询、构造 SGD、生成 |
| `training.py` | 只读评估与最小训练循环、首步快照和记录 |
| `run.py` | 读输入、训练、计数基线对照、频率实验、保存 JSON |
| `check_bigram.py` | 独立数学参考、边界/状态检查、训练与 CLI 验收 |
| `requirements.txt` | 本次使用的 PyTorch 版本 |

建议先看 `TrainableBigram.forward`，再看 `train` 中的五步循环，最后看 `evaluate` 和 `prepare_data`。不要先陷入 CLI 与输出样板。

## 一次运行的数据流

1. `prepare_data(tokenizer, raw)` 检查 train/val 全局唯一文档 id，逐篇编码，每个原文相邻位置只产生一个样本。未知字符注明文档 id 后报错；不连接文档、不使用重叠窗口重复计数。
2. 每组 `PairDataset` 包含文档、样本来源、`x:[B]`、`y:[B]`，两个张量都是 `torch.long`。
3. `TrainableBigram(V)` 注册唯一参数 `score_table:[V,V]`；全零初始化。本模型按行直接学习，不把全零初始化推广到多层网络。
4. `forward(x)` 返回 `score_table[x]`，形状为 `[B,V]`。同一个输入编号不论在哪篇文档或 batch 位置，都查询同一行参数。
5. `make_optimizer` 使用 `model.parameters()`，并检查分数表确实被注册和管理。
6. `train` 每步执行清梯度、前向、交叉熵、backward、step。主实验的所有 6 个训练预测位置共同参与一次更新，真实目标来自训练文档。
7. `evaluate` 仅做前向与稳定交叉熵；不计算新梯度、不清旧梯度、不更新参数。按总位置数平均，空验证组返回 `mean_nll=null`。
8. `generate` 每步查询最后一个 token 的行，选择一个新 token 后追加。采样持续使用局部随机数生成器，不逐步重新播种，也不修改全局 Python 随机状态。

## 实际运行结果

2026-11-13，在上述 `pytorch` 环境使用默认配置运行所得。训练组 6 个预测位置，验证组 2 个预测位置；模型有 9 个可训练标量。

| 模型 | train 平均 NLL | val 平均 NLL |
|---|---:|---:|
| 阶段 1 加一平滑计数模型 | 0.6999508463 | 1.1512925465 |
| 本期初始模型 | 1.0986122887 | 1.0986122887 |
| 本期 300 步后 | 0.3420508021 | 2.2444850618 |

训练损失下降，验证损失上升，是这组数据的真实结果。训练中“春”后只有“风”，本期没有加一平滑，训练后 `P(雨|春)` 从 1/3 降到约 0.01181。但验证文档“春雨风”的第一个目标正是“雨”，该位置 NLL 约 4.43880。另一个验证位置 `P(风|雨)` 约 0.95107，NLL 约 0.05017，两者平均约 2.24449。

不要据此修改验证集或在验证时更新参数，也不把“train 下降”解释成“模型一定泛化更好”。

首步真实记录中，backward 后分数表仍全为 0；SGD 更新后为：

```text
[[-0.03333333,  0.06666667, -0.03333333],
 [ 0.05000000, -0.05000000,  0.00000000],
 [-0.01666667,  0.03333333, -0.01666667]]
```

这是全部 6 个位置平均损失的一次更新，不应与课堂“单个样本”的 `[-0.1,0.2,-0.1]` 强求相同。`training.first_step` 保存了更新前、backward 后、梯度与 step 后四份独立快照。

单独的全覆盖实验用 8 篇双字符文档构成计数 `[[3,1],[1,3]]`，从零参数学习 2000 步；最终概率与未平滑频率 `[[0.75,0.25],[0.25,0.75]]` 的最大绝对误差约 `3.33e-16`，小于验收阈值 0.02。计数频率仅用于比较，没有写入可训练参数。

## 输出怎么看

`output/results.json`：

- `environment` / `sources` / `vocab`：实际环境、输入绝对路径与词表。
- `data`：各组文档、编码、样本来源、x/y 和位置总数。
- `training.config` / `parameter_info`：实际配置、参数清单和数量。
- `training.first_step`：首步参数与梯度快照；主实验 steps=0 时为 null。
- `training.history`：step=0、每 10 步及最终记录，含当时的 logits 和同一时刻重新评估的 train/val 指标。这里 step 表示已完成的更新次数。
- `model`：最终 logits 与逐行概率，列按词表编号排列。
- `final_evaluation`：文档与逐位置明细，含输入、真实目标、logits、目标概率与稳定 NLL。
- `comparison`：同一份数据上的计数、初始可训练、最终可训练模型比较。
- `generations`：argmax 与 sample 的实际文本、编号、提示词与种子。
- `frequency_experiment`：全覆盖实验的数据、计数、目标频率、配置、结果与误差。

`output/checks.json` 保存验收脚本实际输出、检查数量、成功/失败状态与环境。运行失败时 CLI 会在所选输出目录写 `error.json` 并以非零状态退出，包含错误与配置；它不改写上次成功的 `results.json`。旧 error 文件不表示新运行失败，应结合当前退出状态和结果判断。

极小目标概率可能在浮点显示中为 0，但其 NLL 仍按 logits 稳定计算。例如 `[0,-1000]`、目标为第二项，保存的概率可为 0，NLL 为有限的 1000。主模型不通过概率再取 log 来算 loss。

## 验证证据

本阶段 21 个测试方法及其子检查通过，主要覆盖：

- 默认数据与跨文档边界、空组、非法输入、重复 id 和未知字符。
- V=1/2/3/5、B 与 V 不相等的形状、重复输入、参数注册与优化器对象。
- 独立 Python 数学参考的 softmax/交叉熵、平移不变、极大 logits 和极小概率。
- backward 不改参数、SGD 单步手算、清梯度、共享行累积、未查询行不变。
- 验证前后参数和已有梯度保持不变、更换 val 不影响训练轨迹、不等长文档按总位置数平均。
- steps=0、记录时刻、首步快照独立性、NaN/Inf 错误中的步数与配置。
- 默认 300 步训练、全覆盖频率收敛、生成步进/并列/RNG、JSON 保存与 CLI 成败退出。

在相同解释器下，阶段 1 原 18 项检查、0C 原 14 项检查和 0B 原 41 项检查也通过；未修改这些模块。0B 脚本采用原有直接脚本入口：

```bash
python -B learning/stage00_char_codec/check_codec.py
python -B -m learning.stage00_dataset.check_dataset
python -B -m learning.stage01_count_bigram.check_bigram
```

工程自检之外，2026-11-13 正式教学 review 已通过：原 21 项复跑、48 组新数据的独立标量更新对照、已保存 300 步结果重放均通过；没有修改实现。随后实际生成、真实目标评分与验证隔离复盘正确；共享参数为讲解后的新例补验正确。本轮收束，完整解释与待复验项见阶段总结。

## 范围与 AI 使用

本次代码、检查脚本、运行说明与实际运行由 Codex 在用户明确授权后完成；未委派其他 agent。复用项目既有 CharTokenizer、文档校验和 CountBigram 基线，未复制替换旧实现。

参考：任务书中的 Karpathy makemore 神经网络 bigram、CS336 Basics 交叉熵与训练循环主题，以及本 chat 已讲的数值例子。未使用现成模型或 Trainer。当前实现不包含多层网络、GPU 训练、调度器或恢复训练；PyTorch 自动求导的推导将在阶段 3 继续学习。
