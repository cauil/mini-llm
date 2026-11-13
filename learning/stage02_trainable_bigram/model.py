"""模型只存分数：行是当前 token，列是下一 token。"""

import math
import random

import torch
from torch import nn

from learning.stage00_char_codec.tokenizer import CharTokenizer


def require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"{name} 含 NaN/Inf")


class TrainableBigram(nn.Module):
    def __init__(self, vocab_size: int, dtype: torch.dtype = torch.float64):
        super().__init__()
        if type(vocab_size) is not int or vocab_size < 1:
            raise ValueError("词表大小必须是正整数，不能为布尔值")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("本期 dtype 仅支持 float32/float64")
        self.vocab_size = vocab_size
        self.score_table = nn.Parameter(
            torch.zeros(vocab_size, vocab_size, dtype=dtype, device="cpu")
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.dtype != torch.long or x.ndim != 1:
            raise ValueError("x 必须是形状 [B] 的 torch.long 张量")
        if x.device.type != "cpu" or self.score_table.device.type != "cpu":
            raise ValueError("本期仅支持 CPU")
        if x.numel() and ((x < 0).any().item() or (x >= self.vocab_size).any().item()):
            raise ValueError("输入 token 编号越界")
        require_finite("分数表", self.score_table)
        return self.score_table[x]

    @torch.no_grad()
    def distribution(self, current_id: int) -> list[float]:
        if type(current_id) is not int or not 0 <= current_id < self.vocab_size:
            raise ValueError("当前 token 必须是词表内的整数，不能为布尔值")
        logits = self(torch.tensor([current_id], dtype=torch.long))
        return torch.softmax(logits, dim=-1)[0].tolist()

    def snapshot(self) -> list[list[float]]:
        # tolist 产生独立的 Python 数值，不会随后续 step 一起改变。
        return self.score_table.detach().cpu().tolist()

    def parameter_info(self) -> dict:
        parameters = [{"name": name, "shape": list(p.shape),
                       "requires_grad": p.requires_grad, "numel": p.numel()}
                      for name, p in self.named_parameters()]
        return {"parameters": parameters,
                "total_trainable": sum(p["numel"] for p in parameters
                                       if p["requires_grad"])}


def make_optimizer(model: TrainableBigram, learning_rate: float) -> torch.optim.SGD:
    if (type(learning_rate) not in (int, float)
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("学习率必须是有限正数，不能为布尔值")
    parameters = list(model.parameters())
    if len(parameters) != 1 or parameters[0] is not model.score_table:
        raise ValueError("分数表必须是模型唯一注册的参数")
    if not model.score_table.requires_grad:
        raise ValueError("分数表必须允许求梯度")
    optimizer = torch.optim.SGD(parameters, lr=learning_rate,
                                momentum=0, weight_decay=0)
    managed = [p for group in optimizer.param_groups for p in group["params"]]
    if not any(p is model.score_table for p in managed):
        raise ValueError("优化器未管理分数表")
    return optimizer


@torch.no_grad()
def generate(model: TrainableBigram, tokenizer: CharTokenizer, prompt: str,
             num_new_tokens: int, method: str = "sample", seed: int = 0) -> dict:
    if len(tokenizer.vocab) != model.vocab_size:
        raise ValueError("词表大小与模型不一致")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("提示词必须是非空字符串")
    if type(num_new_tokens) is not int or num_new_tokens < 0:
        raise ValueError("新增字符数必须是非负整数，不能为布尔值")
    if method not in ("argmax", "sample"):
        raise ValueError("生成方法必须是 argmax 或 sample")
    if type(seed) is not int:
        raise ValueError("种子必须是整数，不能为布尔值")
    try:
        tokens = tokenizer.encode(prompt)
    except KeyError as error:
        raise ValueError(f"提示词含未知字符：{error.args[0]!r}") from error
    require_finite("分数表", model.score_table)
    rng = random.Random(seed)
    for _ in range(num_new_tokens):
        row = model.distribution(tokens[-1])
        if method == "argmax":
            next_id = max(range(model.vocab_size), key=row.__getitem__)
        else:
            next_id = rng.choices(range(model.vocab_size), weights=row, k=1)[0]
        tokens.append(next_id)
    return {"prompt": prompt, "num_new_tokens": num_new_tokens,
            "method": method, "seed": seed, "tokens": tokens,
            "text": tokenizer.decode(tokens)}
