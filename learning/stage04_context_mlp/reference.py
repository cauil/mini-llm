"""纯 Python/math 参考：独立前向和链式求导，不读取 autograd。

权重布局与 nn.Linear 一致：[输出维度, 输入维度]。
"""

import math


def zeros(rows, columns):
    return [[0.0] * columns for _ in range(rows)]


def affine(x, weight, bias):
    return [[sum(v * w for v, w in zip(row, weights)) + b
             for weights, b in zip(weight, bias)] for row in x]


def reference(parameters: dict, x: list[list[int]], y: list[int]) -> dict:
    E = parameters["embedding.weight"]
    W1, b1 = parameters["layer1.weight"], parameters["layer1.bias"]
    W2, b2 = parameters["layer2.weight"], parameters["layer2.bias"]
    B, k, D, H, V = len(x), len(x[0]), len(E[0]), len(W1), len(E)
    if not B or len(y) != B:
        raise ValueError("参考需要非空且配对的样本")
    emb = [[list(E[token]) for token in row] for row in x]
    z = [[v for vector in row for v in vector] for row in emb]
    a = affine(z, W1, b1)
    h = [[max(0.0, value) for value in row] for row in a]
    logits = affine(h, W2, b2)
    probabilities, losses = [], []
    for row, target in zip(logits, y):
        maximum = max(row)
        exps = [math.exp(value - maximum) for value in row]
        denominator = sum(exps)
        probabilities.append([value / denominator for value in exps])
        losses.append(math.log(denominator) + maximum - row[target])

    gl = [[(p - float(j == y[r])) / B for j, p in enumerate(row)]
          for r, row in enumerate(probabilities)]
    gh, ga, gz = zeros(B, H), zeros(B, H), zeros(B, k * D)
    gE, gW1, gW2 = zeros(V, D), zeros(H, k * D), zeros(V, H)
    gb1, gb2 = [0.0] * H, [0.0] * V
    contributions = []
    for r in range(B):
        for j in range(V):
            gb2[j] += gl[r][j]
            for u in range(H):
                gW2[j][u] += gl[r][j] * h[r][u]
                gh[r][u] += gl[r][j] * W2[j][u]
        for u in range(H):
            ga[r][u] = gh[r][u] if a[r][u] > 0 else 0.0
            gb1[u] += ga[r][u]
            for i in range(k * D):
                gW1[u][i] += ga[r][u] * z[r][i]
                gz[r][i] += ga[r][u] * W1[u][i]
        for position, token in enumerate(x[r]):
            contribution = gz[r][position * D:(position + 1) * D]
            contributions.append({"sample": r, "position": position,
                                  "token_id": token, "gradient": contribution})
            for d, value in enumerate(contribution):
                gE[token][d] += value
    gradients = {"embedding.weight": gE, "layer1.weight": gW1, "layer1.bias": gb1,
                 "layer2.weight": gW2, "layer2.bias": gb2, "z": gz, "a": ga, "h": gh,
                 "logits": gl,
                 "emb": [[row[p * D:(p + 1) * D] for p in range(k)] for row in gz]}
    return {"forward": {"emb": emb, "z": z, "a": a, "h": h, "logits": logits},
            "probabilities": probabilities, "sample_losses": losses,
            "mean_loss": sum(losses) / B, "gradients": gradients,
            "embedding_contributions": contributions}
