"""模型、参数注册、逐步生成与模型重载。"""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from learning.stage00_char_codec.tokenizer import CharTokenizer


def integer(name: str, value: int, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} 必须是 >= {minimum} 的整数，不能为布尔值")


def finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"{name} 含 NaN/Inf")


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int = 2
    embedding_dim: int = 4
    hidden_dim: int = 16
    seed: int = 42

    def __post_init__(self):
        for name in ("vocab_size", "context_length", "embedding_dim", "hidden_dim"):
            integer(name, getattr(self, name))
        integer("seed", self.seed, 0)
        if self.seed >= 2**63:
            raise ValueError("seed 必须小于 2**63")


class ContextMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # 局部初始化不改变调用者的随机数状态。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim,
                                          dtype=torch.float64, device="cpu")
            self.layer1 = nn.Linear(config.context_length * config.embedding_dim,
                                    config.hidden_dim, dtype=torch.float64, device="cpu")
            self.layer2 = nn.Linear(config.hidden_dim, config.vocab_size,
                                    dtype=torch.float64, device="cpu")
            with torch.no_grad():
                self.embedding.weight.normal_(0, 0.1)
                self.layer1.bias.zero_()
                self.layer2.bias.zero_()

    def features(self, x: torch.Tensor, *, detach_hidden: bool = False,
                 retain: bool = False) -> dict[str, torch.Tensor]:
        if (not isinstance(x, torch.Tensor) or x.dtype != torch.long
                or x.device.type != "cpu" or x.ndim != 2
                or x.shape[0] == 0 or x.shape[1] != self.config.context_length):
            raise ValueError("x 必须是非空 CPU long 张量 [B,k]，k 匹配模型")
        if ((x < 0) | (x >= self.config.vocab_size)).any().item():
            raise ValueError("输入 token 编号越界")
        for name, parameter in self.named_parameters():
            if parameter.device.type != "cpu" or parameter.dtype != torch.float64:
                raise ValueError("本期参数必须为 CPU float64")
            finite(name, parameter)
        emb = self.embedding(x)
        z = emb.reshape(x.shape[0], -1)
        a = self.layer1(z)
        h = torch.relu(a)
        if detach_hidden:
            h = h.detach()
        logits = self.layer2(h)
        result = {"emb": emb, "z": z, "a": a, "h": h, "logits": logits}
        for name, value in result.items():
            finite(name, value)
            if retain and value.requires_grad:
                value.retain_grad()
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)["logits"]

    def parameter_info(self) -> dict:
        entries = [{"name": name, "shape": list(p.shape), "numel": p.numel(),
                    "requires_grad": p.requires_grad} for name, p in self.named_parameters()]
        return {"parameters": entries,
                "total_trainable": sum(e["numel"] for e in entries if e["requires_grad"])}


def snapshot(model: nn.Module, *, gradients: bool = False) -> dict:
    result = {}
    for name, parameter in model.named_parameters():
        value = parameter.grad if gradients else parameter
        result[name] = None if value is None else value.detach().tolist()
    return result


@torch.no_grad()
def generate(model: ContextMLP, tokenizer: CharTokenizer, prompt: str,
             num_new_tokens: int, *, method: str = "sample", seed: int = 42) -> dict:
    integer("num_new_tokens", num_new_tokens, 0)
    integer("seed", seed, 0)
    if seed >= 2**63:
        raise ValueError("seed 必须小于 2**63")
    if method not in ("sample", "argmax"):
        raise ValueError("method 必须为 sample 或 argmax")
    if len(tokenizer.vocab) != model.config.vocab_size:
        raise ValueError("词表大小与模型不匹配")
    if not isinstance(prompt, str):
        raise ValueError("prompt 必须为字符串")
    try:
        tokens = tokenizer.encode(prompt)
    except KeyError as error:
        raise ValueError(f"提示词含未知字符：{error.args[0]!r}") from error
    k = model.config.context_length
    if len(tokens) < k:
        raise ValueError(f"提示词必须至少包含 {k} 个字符")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    was_training = model.training
    model.eval()
    trace = []
    try:
        for _ in range(num_new_tokens):
            context = tokens[-k:]
            probabilities = model(torch.tensor([context], dtype=torch.long)).softmax(-1)[0]
            token = (int(probabilities.argmax()) if method == "argmax" else
                     int(torch.multinomial(probabilities, 1, generator=generator)))
            trace.append({"context_ids": list(context),
                          "context_text": tokenizer.decode(context),
                          "probabilities": probabilities.tolist(),
                          "next_id": token, "next_text": tokenizer.decode([token])})
            tokens.append(token)
    finally:
        model.train(was_training)
    return {"prompt": prompt, "method": method, "seed": seed,
            "num_new_tokens": num_new_tokens, "trace": trace,
            "token_ids": tokens, "text": tokenizer.decode(tokens)}


def save_checkpoint(path: Path, model: ContextMLP, tokenizer: CharTokenizer) -> None:
    if len(tokenizer.vocab) != model.config.vocab_size:
        raise ValueError("checkpoint 词表与模型大小不匹配")
    for name, parameter in model.named_parameters():
        finite(name, parameter)
    torch.save({"format_version": 1, "config": asdict(model.config),
                "vocab": tokenizer.vocab, "state_dict": model.state_dict()}, path)


def load_checkpoint(path: Path) -> tuple[ContextMLP, CharTokenizer]:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format_version") != 1:
        raise ValueError("不支持的 checkpoint 格式")
    config = ModelConfig(**saved["config"])
    tokenizer = CharTokenizer(saved["vocab"])
    if config.vocab_size != len(tokenizer.vocab):
        raise ValueError("checkpoint 词表与配置不匹配")
    model = ContextMLP(config)
    model.load_state_dict(saved["state_dict"], strict=True)
    for name, parameter in model.named_parameters():
        finite(name, parameter)
    return model, tokenizer
