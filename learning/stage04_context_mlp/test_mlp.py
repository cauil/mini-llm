"""额外行为测试：新配置梯度、评估状态、快照、CLI及失败出口。

全部文件写入临时目录，不覆盖 output/。
"""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from learning.stage00_char_codec.tokenizer import CharTokenizer
from .checks import Checks
from .data import prepare_data
from .model import ContextMLP, ModelConfig, generate, load_checkpoint, save_checkpoint, snapshot
from .reference import reference
from .run import ROOT, execute, parser
from .training import evaluate, train


class ContextMLPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_reference_on_new_configurations_and_labels(self):
        for seed in range(12):
            V, k, D, H, B = 2 + seed % 4, 1 + seed % 3, 1 + seed % 4, 3 + seed % 3, 1 + seed % 5
            with self.subTest(seed=seed, V=V, k=k, D=D, H=H, B=B):
                model = ContextMLP(ModelConfig(V, k, D, H, seed))
                generator = torch.Generator().manual_seed(seed + 500)
                x = torch.randint(V, (B, k), generator=generator)
                for shift in (0, 1):
                    y = (torch.arange(B) + shift) % V
                    model.zero_grad(set_to_none=True)
                    f = model.features(x, retain=True)
                    expected = reference(snapshot(model), x.tolist(), y.tolist())
                    loss = F.cross_entropy(f["logits"], y)
                    loss.backward()
                    torch.testing.assert_close(loss.detach(), torch.tensor(expected["mean_loss"], dtype=torch.float64))
                    actual = dict(model.named_parameters()) | f
                    for name, values in expected["gradients"].items():
                        torch.testing.assert_close(actual[name].grad, torch.tensor(values, dtype=torch.float64),
                                                   atol=1e-10, rtol=1e-8)

    def test_rng_seed_and_zero_step_snapshot(self):
        before_rng = torch.random.get_rng_state().clone()
        model = ContextMLP(ModelConfig(4, 2, 3, 5, 37))
        self.assertTrue(torch.equal(before_rng, torch.random.get_rng_state()))
        other = ContextMLP(model.config)
        self.assertEqual(snapshot(model), snapshot(other))
        initial = snapshot(model)
        record = train(model, torch.tensor([[0, 1]]), torch.tensor([2]), steps=0)
        self.assertEqual(snapshot(model), initial)
        self.assertIsNone(record["first_step"])
        self.assertEqual(len(record["history"]), 1)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_eval_generation_preserve_state_and_weighted_average(self):
        model = ContextMLP(ModelConfig(4, 2, 3, 5))
        tokenizer = CharTokenizer({"春": 0, "风": 1, "雨": 2, "山": 3})
        x = torch.tensor([[0, 1], [1, 2], [2, 0], [3, 0], [3, 3]])
        y = torch.tensor([2, 0, 1, 1, 0])
        F.cross_entropy(model(x), y).backward()
        parameters, gradients = snapshot(model), snapshot(model, gradients=True)
        with torch.no_grad():
            expected = sum(float(F.cross_entropy(model(x[i:i+1]), y[i:i+1])) for i in range(len(y))) / len(y)
        for training in (True, False):
            model.train(training)
            for batch_size in (1, 2, 3, 8):
                result = evaluate(model, x, y, batch_size=batch_size)
                self.assertAlmostEqual(result["loss"], expected, places=12)
                self.assertEqual(model.training, training)
            sample = generate(model, tokenizer, "春风雨", 7, method="sample", seed=17)
            self.assertEqual(sample, generate(model, tokenizer, "春风雨", 7, method="sample", seed=17))
            self.assertEqual(model.training, training)
        self.assertEqual(parameters, snapshot(model))
        self.assertEqual(gradients, snapshot(model, gradients=True))

    def test_snapshots_and_checkpoint_are_independent(self):
        model = ContextMLP(ModelConfig(3, 2, 2, 4))
        tokenizer = CharTokenizer({"春": 0, "风": 1, "雨": 2})
        x, y = torch.tensor([[0, 1], [1, 2]]), torch.tensor([2, 0])
        record = train(model, x, y, steps=3, log_every=2)
        frozen_record = copy.deepcopy(record)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            save_checkpoint(path, model, tokenizer)
            restored, restored_tokenizer = load_checkpoint(path)
            self.assertEqual(restored_tokenizer.vocab, tokenizer.vocab)
            torch.testing.assert_close(restored(x), model(x), atol=0, rtol=0)
            with torch.no_grad():
                model.layer2.bias.add_(1)
            self.assertFalse(torch.equal(restored(x), model(x)))
            self.assertEqual(record, frozen_record)

    def test_data_new_k_and_failure_does_not_mutate_source(self):
        raw = {"train": [{"id": "a", "text": "春风雨春风"}, {"id": "b", "text": "风"}],
               "val": [{"id": "v", "text": "雨春风雨"}]}
        tokenizer = CharTokenizer({"春": 0, "风": 1, "雨": 2})
        original = copy.deepcopy(raw)
        for k in (1, 2, 3, 5):
            groups, _ = prepare_data(raw, tokenizer, k)
            self.assertEqual(groups["train"].x.shape, (max(0, 5-k), k))
            for sample in groups["train"].samples:
                start = sample["start"]
                self.assertEqual(sample["context_ids"], tokenizer.encode(raw["train"][0]["text"][start:start+k]))
                self.assertEqual(sample["target_id"], tokenizer.encode(raw["train"][0]["text"][start+k])[0])
        self.assertEqual(raw, original)

    def test_bad_labels_nonfinite_and_disconnected_training_fail(self):
        model = ContextMLP(ModelConfig(3, 2, 2, 4))
        x = torch.tensor([[0, 1], [1, 2]])
        for bad in (torch.tensor([0.0, 1.0]), torch.tensor([0]), torch.tensor([-1, 0]), torch.tensor([3, 0])):
            with self.subTest(labels=bad), self.assertRaises(ValueError):
                train(model, x, bad, steps=1)
        y = torch.tensor([0, 1])
        hook = model.layer2.bias.register_hook(lambda grad: torch.full_like(grad, float("nan")))
        with self.assertRaises(FloatingPointError):
            train(model, x, y, steps=1)
        hook.remove()
        with patch.object(model, "forward", side_effect=lambda inputs: model.features(inputs, detach_hidden=True)["logits"]):
            with self.assertRaisesRegex(RuntimeError, "未收到梯度"):
                train(model, x, y, steps=1)

    def test_failed_check_is_saved_and_rejected(self):
        checks = Checks()
        checks.record("injected_failure", False)
        with tempfile.TemporaryDirectory() as folder:
            args = parser().parse_args(["--output-dir", folder, "--steps", "0", "--num-new-tokens", "0"])
            with patch("learning.stage04_context_mlp.run.run_checks", return_value=checks):
                with self.assertRaisesRegex(RuntimeError, "injected_failure"):
                    execute(args)
            report = json.loads((Path(folder) / "checks.json").read_text())
            self.assertFalse(report["all_passed"])
            self.assertEqual(report["failed"], 1)

    def test_cli_alternate_dimensions_reload_in_new_process_and_failure_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            command = [sys.executable, "-m", "learning.stage04_context_mlp.run",
                       "--output-dir", folder, "--context-length", "3", "--embedding-dim", "2",
                       "--hidden-dim", "5", "--steps", "3", "--log-every", "2", "--num-new-tokens", "2"]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads((Path(folder) / "results.json").read_text())
            self.assertEqual(result["config"]["context_length"], 3)
            self.assertEqual([r["step"] for r in result["training"]["history"]], [0, 2, 3])
            # 父进程加载子进程产物，检查参数可跨进程复现。
            model, _ = load_checkpoint(Path(folder) / "model.pt")
            logits = model(torch.tensor(result["data"]["train"]["x"]))
            torch.testing.assert_close(logits, torch.tensor(result["checkpoint"]["original_logits"], dtype=torch.float64),
                                       atol=0, rtol=0)
            before = (Path(folder) / "results.json").read_bytes()
            invalid = subprocess.run(command + ["--learning-rate", "nan"], cwd=ROOT,
                                     capture_output=True, text=True, timeout=20)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertEqual((Path(folder) / "results.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
