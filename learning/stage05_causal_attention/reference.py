"""慢速独立参考：只遍历允许读取的前缀，不构造/复用向量化mask。"""

import math
import torch


def project_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """weight沿用nn.Linear的[输出,输入]存储，逐样本/位置/输出特征点积。"""
    return torch.stack([
        torch.stack([torch.stack([torch.dot(token, row) for row in weight])
                     for token in sample])
        for sample in x
    ])


def attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """仅接受已验证形状的教学输入，返回(H,A)；保留autograd路径。"""
    outputs, weights = [], []
    for b in range(q.shape[0]):
        sample_h, sample_a = [], []
        for i in range(q.shape[1]):
            scores = torch.stack([
                torch.dot(q[b, i], k[b, j]) / math.sqrt(q.shape[-1])
                for j in range(i + 1)
            ])
            # 独立写出稳定归一化，避免共享主实现的softmax维度错误。
            numerators = torch.exp(scores - scores.max())
            allowed = numerators / numerators.sum()
            row_h = torch.zeros_like(v[b, 0])
            for j in range(i + 1):
                row_h = row_h + allowed[j] * v[b, j]
            sample_h.append(row_h)
            sample_a.append(torch.cat((allowed, allowed.new_zeros(q.shape[1] - i - 1))))
        outputs.append(torch.stack(sample_h))
        weights.append(torch.stack(sample_a))
    return torch.stack(outputs), torch.stack(weights)
