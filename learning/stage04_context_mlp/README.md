# 阶段 4：上下文 MLP

按[任务书 v1](/Users/lcf/llm/tasks/阶段4_上下文MLP任务书.md)实现。用户于2026-11-16明确授权本 chat 编码；2026-11-17正式工程review通过，见[审查记录](/Users/lcf/llm/reviews/阶段4_代码审查.md)。核心代码复盘及本轮G1已收束，总结见[04-embedding与MLP.md](/Users/lcf/llm/04-embedding与MLP.md)，核心流程仍可查[伪代码讲义](/Users/lcf/llm/reviews/阶段4_核心流程复盘.md)。用户最新要求撤回重复的G1编码任务，采用既有工程与课堂迁移证据；以下运行结果不替代用户答题，提示/纠正边界保留。

## 运行

使用已有环境，无需新装框架。所有命令从项目根目录执行：

```bash
cd /Users/lcf/llm
/Users/lcf/.conda/envs/pytorch/bin/python -m learning.stage04_context_mlp.run
```

完整入口执行默认文档训练、核心检查、合成对照、生成、checkpoint保存/重载与JSON读回校验；覆盖本目录 `output/` 中同名产物。约数秒，CPU单线程；不会改写旧阶段输出。

仅运行核心正确性检查，不训练、不写正式输出：

```bash
/Users/lcf/.conda/envs/pytorch/bin/python -m learning.stage04_context_mlp.check_mlp
```

额外行为测试（写入自动清理的临时目录）：

```bash
/Users/lcf/.conda/envs/pytorch/bin/python -m unittest learning.stage04_context_mlp.test_mlp -v
```

形状迁移示例，使用独立输出目录：

```bash
/Users/lcf/.conda/envs/pytorch/bin/python -m learning.stage04_context_mlp.run \
  --context-length 3 --embedding-dim 2 --hidden-dim 5 \
  --steps 3 --log-every 2 --num-new-tokens 2 \
  --output-dir /private/tmp/stage04-k3
```

`--documents/--vocab`可指定同格式文档/词表；`--steps/--seed/--learning-rate`等控制主模型。`--steps 0`只跳过主模型更新；合成实验仍使用独立固定配置，并执行所有检查。若默认文档长度不足新k导致整个训练组无样本，会明确报错，不自动补padding。

核心检查失败、非有限值或其他未预期异常均导致非零退出。完整运行的普通数值检查失败会保存 `checks.json` 中的失败项后报错；更早的输入/运行异常直接在stderr报告，不伪造成功结果。运行前已有output时，早期失败可能留下上一次产物，应以本次退出状态为准。

## 代码入口

| 文件 | 责任 |
|---|---|
| `model.py` | ModelConfig、参数初始化、embedding/两层Linear、前向中间量、生成、checkpoint |
| `data.py` | 只读调用阶段0C，把后移序列适配为一个目标，并保留样本来源 |
| `training.py` | 全量batch SGD、按目标数评估、首步快照、有限值检查 |
| `reference.py` | 纯Python/math独立前向与解析反向，不读取autograd |
| `checks.py` | A/B/C检查、梯度证据、边界拒绝、detach和初始化对照 |
| `run.py` | 默认训练、合成MLP/bigram对照、生成和产物保存 |
| `check_mlp.py` | 核心148项检查的独立入口 |
| `test_mlp.py` | 8个行为测试方法，含24组新配置/标签梯度子例、进程与失败出口 |

## 数据与模型接口

默认只读阶段0C的 `documents.json` 和 `vocab.json`，直接复用 `CharTokenizer` 及 `prepare_dataset(T=k)`，不消费旧T=3输出。词表为春/风/吹/过/山/河，V=6。

默认k=2：训练文档“春风吹过山河”得到4条样本；验证文档“山河春风”得到2条样本。例如输入“春风”，目标“吹”。每篇按原文起点逐个取窗口，目标取0C窗口后移标签的最后一项；不跨文档连接或随机重划分。

样本包含 `split/doc_id/start/context_ids/target_id/target_position/context_text/target_text`。结果同时保存源文件绝对路径与SHA256、原文、词表、零样本文档的metadata。未知字符即使出现在短文档也会报错；空验证组 `loss=null`，空训练组拒绝。

模型输入 `x:[B,k]` 为CPU long，目标 `y:[B]`；参数为CPU float64。前向形状：

```text
x [B,k]
→ embedding [B,k,D]
→ reshape [B,kD]（保留batch，保留位置顺序）
→ layer1 [B,H]
→ ReLU [B,H]
→ layer2/logits [B,V]
```

`nn.Linear.weight`按 `[输出维,输入维]` 存储。参数总数为：

`V×D + H×(kD) + H + V×H + V`。

默认V=6、k=2、D=4、H=16，共270个可训练标量。E与输出权重不共享，所有参数均注册并由优化器管理。默认E随机标准差0.1、Linear权重采用PyTorch默认初始化、偏置为0；局部seed初始化不改变调用者的全局随机数状态。

## 训练与生成约定

- 主训练默认seed42、lr0.1、1000步，每50步记录；CPU单线程、无动量/权重衰减的SGD。
- 每步使用全部train样本；损失平均分母为预测目标数B。评估可以分批，但先累加NLL再除以总目标数；不等长末批不会多占权重。
- 首步保存before、after_backward、gradients、after_step独立副本，检验backward不更新参数以及SGD更新式。
- 训练期任何参数缺失梯度均作为异常；梯度为数值0不等于缺失。诊断实验中预期None单独记录，不送入正常训练器。
- 生成要求提示词至少k个字符；每步取最新末尾k个token、得到概率、采样/argmax并追加。固定seed的采样可重放，记录每步实际上下文和分布。
- 保存模型配置、词表与state_dict；以 `weights_only=True` 加载到全新模型，核对logits完全一致。此checkpoint不含优化器状态，不用于续训恢复。

## 梯度证据与诊断

`checks.json → experiments.gradients`保存两组固定例子的输入、参数、真实各层前向值、概率/损失、独立解析结果、autograd梯度，以及每个位置对embedding行的贡献。

参考只使用Python列表和math进行计算，不调用autograd；按平均CE先缩放各样本logit梯度，再通过两个Linear、ReLU和按位置拆分/共享行求和得到参数梯度。覆盖重复token、同一batch多次使用一行、未查询行、bias按样本维求和。另检查loss乘2时所有参数梯度相应乘2。

中间量使用 `retain_grad()` 观察；未保留非叶.grad不代表路径断开。解析验导数使用避开0的ReLU输入；初始化全零诊断有意包含0，不把该诊断当作拐点导数检验。

`experiments.detach`：正常路径与hidden.detach路径的logits/loss一致；detach阻断E/第一层，第二层仍有梯度。`experiments.initialization`：相同E下比较随机Linear与全零Linear；全零时只有输出偏置得到非零梯度。诊断使用独立模型，不污染主训练。

本期没有重新手写或要求掌握差分实现。两个固定例子允许改变标签；额外测试以12组不同尺寸/种子、各2个标签配置验证解析参考不是硬编码答案。

## 实际运行结果（2026-11-16）

环境：Python3.12.16、PyTorch2.15.0、CPU、float64、单线程。完整入口232/232项通过；核心检查148/148项通过；额外8个测试方法全部通过。额外测试包括24组新配置/标签、batch加权、状态保护、零步训练、快照独立性、跨进程checkpoint、不同k的CLI、注入检查失败/断梯度/NaN后的失败出口。

固定多层解析与autograd最大绝对梯度误差约 `3.47e-17`，容差atol=1e-8、rtol=1e-6。保存重载logits完全一致。

| 默认文档指标 | 初始化 | 1000步后 |
|---|---:|---:|
| train平均CE（4个目标） | 1.7975797392 | 0.0010319304 |
| val平均CE（2个目标） | 1.7988766664 | 4.0501918355 |

训练拟合而验证恶化，符合这个极小、上下文覆盖不足数据上的过拟合现象；不宣称泛化改善。真实生成样例也主要重复训练片段，不能作为语言质量验收依据。

独立合成实验只用“我爱猫”“你爱狗”两篇，每篇一个目标，无验证集。两模型比较完全相同的目标；bigram仅输入末字“爱”，MLP输入“我爱/你爱”。MLP默认1000步首次即达标，无调参重试：

| 模型 | 平均CE | 两上下文目标概率 |
|---|---:|---|
| MLP | 0.0005284142 | 猫0.9994665；狗0.9994770 |
| bigram | 0.7056657883 | 两上下文分布完全相同，详见JSON |

这是已见样本的拟合能力对照，不是独立泛化证据。运行器为此实验只预设1000/3000步两次有界尝试，每次从相同初值重建，并保存所有实际尝试及原因；两次均不达标时失败，不无限训练。

## 产物与提交边界

- `output/results.json`：主配置/环境、源数据、样本、训练曲线、首步快照、合成实验、生成过程、checkpoint核对。
- `output/checks.json`：232项实际检查、参考误差、预期异常及完整梯度/诊断证据。
- `output/model.pt`：最终主模型、配置和词表。

实现中明确区分两种标签接口、两种权重布局和两种None来源；没有修改旧阶段接口来迁就新任务。默认配置首次运行已达合成验收要求，验证集恶化如实保留。依赖 `requirements.txt` 只列PyTorch，具体复现版本以上述环境记录为准。

AI使用范围：本chat在用户明确授权后编写本模块、检查与文档并运行实验，未委派其他agent。参考本地任务书、课堂推导和已有阶段0/2接口，没有复制外部完整模型或训练器。课程参考见任务书中的Karpathy/CS336入口。

2026-11-17正式review：原8个测试方法复跑通过，临时目录完整入口232项通过，两份输出JSON与原提交严格一致；新增独立50位Decimal前向模式求导与行为复验302/302项通过，24组新网络的参数梯度最大误差约2.22e-16。原Python/输出共12文件哈希不变，未修复实现。

2026-11-17阶段收束：核心代码复盘、课堂新文本/新k及断梯度检查已有证据，阶段总结已合并三份side chat。新增G1编码草案已撤回且未执行；既有工程结果保持，不因文档总结重新训练。下一阶段为单头因果注意力。
