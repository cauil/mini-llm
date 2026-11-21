"""输入特征 → 共享投影 → 单头因果注意力。诊断结果保留计算图。"""

from dataclasses import dataclass
import math

import torch
from torch import nn


def positive_integer(name: str, value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} 必须为正整数（不接受bool）")


def finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"{name} 含NaN/Inf或计算溢出")


def features(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} 必须是Tensor")
    if tensor.ndim != 3 or any(size == 0 for size in tensor.shape):
        raise ValueError(f"{name} 必须是非空三维张量 [B,T,特征维]")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} 必须是实数浮点张量")


@dataclass(frozen=True)
class AttentionConfig:
    d_model: int
    d_key: int
    d_value: int
    seed: int = 42

    def __post_init__(self):
        for name in ("d_model", "d_key", "d_value"):
            positive_integer(name, getattr(self, name))
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed 必须是[0,2**63)内的整数（不接受bool）")


@dataclass(frozen=True)
class AttentionTrace:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    scores: torch.Tensor
    future_mask: torch.Tensor
    masked_scores: torch.Tensor
    weights: torch.Tensor
    output: torch.Tensor


def attention_trace(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> AttentionTrace:
    """核心计算；同批、同长度的自注意力。mask为[T,T]，其余保留batch。"""
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        features(name, tensor)
    if q.shape != k.shape or q.shape[:2] != v.shape[:2]:
        raise ValueError("Q/K须同形状[B,T,d_k]，V须匹配B/T")
    if any(t.dtype != q.dtype or t.device != q.device for t in (k, v)):
        raise ValueError("Q/K/V的dtype与device必须相同")
    for name, value in (("q", q), ("k", k), ("v", v)):
        finite(name, value)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    finite("scores", scores)
    length = q.shape[1]
    future_mask = torch.ones(length, length, dtype=torch.bool, device=q.device).triu(1)
    masked_scores = scores.masked_fill(future_mask, -torch.inf)
    weights = torch.softmax(masked_scores, dim=-1)
    output = weights @ v
    finite("weights", weights)
    finite("output", output)
    return AttentionTrace(q, k, v, scores, future_mask, masked_scores, weights, output)


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """返回(H,A)，形状分别为[B,T,d_v]与[B,T,T]。"""
    trace = attention_trace(q, k, v)
    return trace.output, trace.weights


class SingleHeadCausalAttention(nn.Module):
    def __init__(self, config: AttentionConfig):
        super().__init__()
        if not isinstance(config, AttentionConfig):
            raise TypeError("config 必须为AttentionConfig")
        self.config = config
        # 使用Linear默认初始化，同时还原调用者的CPU随机数状态。
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            options = dict(bias=False, dtype=torch.float64, device="cpu")
            self.query = nn.Linear(config.d_model, config.d_key, **options)
            self.key = nn.Linear(config.d_model, config.d_key, **options)
            self.value = nn.Linear(config.d_model, config.d_value, **options)

    def project(self, x: torch.Tensor):
        features("x", x)
        if x.shape[-1] != self.config.d_model:
            raise ValueError("x最后一维必须等于d_model")
        for name, parameter in self.named_parameters():
            if parameter.dtype != x.dtype or parameter.device != x.device:
                raise ValueError(f"x与参数{name}的dtype/device不匹配")
            finite(name, parameter)
        finite("x", x)
        return self.query(x), self.key(x), self.value(x)

    def inspect(self, x: torch.Tensor) -> AttentionTrace:
        """与forward使用同一计算路径；调用者仅在保存日志时detach。"""
        return attention_trace(*self.project(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inspect(x).output
