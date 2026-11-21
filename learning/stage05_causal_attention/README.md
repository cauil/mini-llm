# 阶段 5：单头因果注意力

实现日期：2026-11-21。依据[任务书v1](/Users/lcf/llm/tasks/阶段5_单头因果注意力任务书.md)。用户明确授权本chat实现；正式教学review通过，本轮核心代码复盘收束。见[审查报告](/Users/lcf/llm/reviews/阶段5_代码审查.md)及[核心流程复盘](/Users/lcf/llm/reviews/阶段5_核心流程复盘.md)。

## 运行

从 `/Users/lcf/llm` 执行，使用已有环境，无需重新安装PyTorch：

```bash
/Users/lcf/.conda/envs/pytorch/bin/python -B -m learning.stage05_causal_attention.check_attention
/Users/lcf/.conda/envs/pytorch/bin/python -B -m unittest learning.stage05_causal_attention.test_attention -v
```

检查入口默认写本目录的`output/checks.json`和`output/example.json`，可以用`--output-dir /absolute/path`指定独立输出目录。正确实现不通过或故障变体未被检出时返回退出码1；报告保留真实失败项。

实测环境：Python 3.12.16、PyTorch 2.15.0、CPU、float64、检查入口单线程。requirements.txt记录实测版本，不要求重装现有环境。其他设备/精度未做工程验证。

## 文件与接口

| 文件 | 作用 |
|---|---|
| [model.py](/Users/lcf/llm/learning/stage05_causal_attention/model.py) | 三组共享投影、手写注意力核心、输入校验、可观察中间值 |
| [reference.py](/Users/lcf/llm/learning/stage05_causal_attention/reference.py) | 逐样本/位置/特征投影；只访问前缀的独立循环参考 |
| [diagnostics.py](/Users/lcf/llm/learning/stage05_causal_attention/diagnostics.py) | 隔离的三种故障变体，正式模型不导入它 |
| [check_attention.py](/Users/lcf/llm/learning/stage05_causal_attention/check_attention.py) | 任务v1自检与真实JSON报告 |
| [test_attention.py](/Users/lcf/llm/learning/stage05_causal_attention/test_attention.py) | 非连续输入、梯度对照、状态保护、设备错误与失败出口测试 |

用法：

```python
import torch
from learning.stage05_causal_attention import AttentionConfig, SingleHeadCausalAttention, causal_attention

model = SingleHeadCausalAttention(AttentionConfig(d_model=5, d_key=3, d_value=2, seed=42))
x = torch.randn(2, 4, 5, dtype=torch.float64)
h = model(x)                       # [2,4,2]
trace = model.inspect(x)            # 与forward同一计算路径，保留计算图
assert trace.weights.shape == (2, 4, 4)
h2, a2 = causal_attention(trace.q, trace.k, trace.v)
```

- `forward(X)`返回H，`inspect(X)`返回AttentionTrace，包含q/k/v、缩放后的scores、布尔future_mask、masked_scores、weights和output。
- 核心`causal_attention(Q,K,V)`返回`(H,A)`。Q/K必须为相同的`[B,T,d_key]`，V为`[B,T,d_value]`，不允许偷偷广播不同样本或序列长度。
- 输入X为`[B,T,d_model]`；参数默认CPU/float64，输入必须与参数dtype/device匹配。拒绝空维度、整型/复数、rank/维度错误、非有限值及运算溢出。配置尺寸为正整数，不接受bool。
- T不属于可训练参数配置。同一实例能处理不同T，每次根据本次T生成上三角mask，且保留对角线。
- 三组`nn.Linear(..., bias=False)`使用PyTorch默认初始化。构造时用配置seed固定CPU随机数，并还原调用者的CPU RNG状态；无dropout，forward不消耗随机数。

参数名为`query.weight`、`key.weight`、`value.weight`。PyTorch存储形状分别为`[d_key,D]`、`[d_key,D]`、`[d_value,D]`，实际前向乘权重转置。加载课堂数学布局`[D,输出维]`的矩阵时需要转置。参数总量为`D*(2*d_key+d_value)`；所有位置及样本共享这些参数。Q/K/V、A、H不注册为参数。

## 计算流程

```text
X [B,T,D]
→ 三组无偏置Linear
→ Q/K [B,T,d_key]，V [B,T,d_value]
→ Q @ K.transpose(-2,-1) / sqrt(d_key) [B,T,T]
→ 将列号大于行号的位置填为负无穷
→ softmax(dim=-1)，固定样本/查询行，对被读取位置归一化
→ A @ V，得到H [B,T,d_value]
```

循环参考不构造这个mask，而是只循环到当前查询位置；每个分数单独做点积，归一化独立写为减最大值、指数、除总和，再逐位置加权求和。它没有调用主实现或复用主实现的权重矩阵。

## 实际结果

[checks.json](/Users/lcf/llm/learning/stage05_causal_attention/output/checks.json)记录141/141项检查通过；[example.json](/Users/lcf/llm/learning/stage05_causal_attention/output/example.json)保存课堂案例的完整中间值和PyTorch布局参数。JSON中被遮分数使用字符串`"-inf"`，没有非标准Infinity/NaN数值。

- 课堂输出H为`[[[1,0],[1/3,2/3],[5/9,8/9]]]`，与用户手算一致。
- 3组新配置覆盖B=1/B>1、T=1/T>1、d_key>1和d_key不等于d_value；投影、权重与输出的独立参考最大绝对误差为`2.220446049250313e-16`。容差为`atol=1e-8, rtol=1e-6`。
- 同一实例支持T=1、3、5；逐样本与batch结果一致，修改一个样本不影响其他样本，输入及参数保持原值。
- 对B=2、T=4的固定非退化输入，用0基截止位置c=0和c=1分别修改未来X，重新计算整个模块。两次正确前缀输出最大误差均为0，未来投影确实发生变化。

同样的扰动在故障变体下产生以下前缀输出误差（都是有限数值）：

| 故障 | c=0 | c=1 |
|---|---:|---:|
| 移除mask | 1.372603 | 1.054057 |
| 反转mask方向、保留自己 | 1.372603 | 1.240847 |
| 沿查询轴softmax | 0.031741 | 0.076404 |

三种故障都被同一个前缀不变检查检出。错误轴变体的未来权重仍为0，但行和及因果性检查失败；未来查询通过归一化分母影响了前面的输出。因此零未来权重不能单独作为因果性证明。

诊断标量`L=(H*R).sum()`的backward验证了输入与三组参数具有形状正确、有限且非全零的梯度；未运行optimizer或训练。141项是检查项数，不是141次独立实验。

补充的6个unittest方法全部通过：

- 两组新尺寸与非连续输入的前向、X及三组参数梯度，对照独立循环参考的autograd结果，容差`atol=1e-10, rtol=1e-8`。
- 初始化与前向不改变调用者CPU RNG、输入、参数或已有梯度，train/eval前向一致。
- 用meta张量模拟不匹配设备时获得明确校验错误，无需GPU。
- 约1000的有限分数验证稳定softmax，输出与参考一致。
- 给正式计算临时注入真实的漏mask错误，主检查入口返回1并记录失败。
- 把故障变体临时替换为正确实现，检查入口发现“故障未被检出”并返回1，避免把预期失败写死。

上述两个注入测试的输出只写临时目录，不污染默认报告。正式输出由干净的完整检查入口再次生成。源码编译及`git diff --check`通过；执行前后旧阶段53个非缓存文件的SHA256一致。

## 实现复盘与边界

代码检查时发现，若先检查张量数值是否有限、再比较设备，会在无实际存储的meta设备上先遇到`.item()`错误。实现改为先核对形状、dtype/device，再读取数值；补充测试验证能明确报告设备不匹配。

生产计算完整保留autograd。报告序列化才使用detach并复制为Python数值，不会通过日志切断正式计算路径。反向连通性不等同完整求导证明；补充梯度测试是有限新例与独立循环图的对照，不是手推attention反向。

本期没有新增训练、文本生成、多头、位置编码、输出投影、padding或KV cache。模块可在阶段6扩展，当前验收范围是任务v1；没有泛化或性能结论。

用户已授权本chat编写全部阶段5源码、检查及说明，未委派其他agent。实现依据课堂公式和任务规格；课程参考为[Karpathy GPT主题](https://karpathy.ai/zero-to-hero.html)与[CS336 Basics](https://github.com/stanford-cs336/assignment1-basics)，未复制现成注意力或完整模型实现。

2026-11-21正式review：原141项及6个测试方法复跑通过，两份JSON与原提交全部字段一致；新增Python math标量前向/解析反向复验118/118项通过（8个新网络），前向最大误差约3.89e-16、梯度约2.22e-16，被审源码/依赖/原产物9文件SHA256不变。后续核心代码理解复盘已本轮收束，提示/纠正边界见[05-单头因果注意力.md](/Users/lcf/llm/05-单头因果注意力.md)；该总结已合并主chat与side chat 1/2，不等于全部无提示或延迟验收通过。
