"""额外验证：新输入、差分不变量与失败出口，不重复六组固定实验。"""

import contextlib
import io
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from .core import Checks, DTYPE, central_difference, compare_sample, reference
from . import run


class GradientChecks(unittest.TestCase):
    def test_unseen_logits_and_targets(self):
        rng = random.Random(20261115)
        for vocab_size in (1, 2, 3, 5, 7):
            for _ in range(4):
                scores = [rng.uniform(-3, 3) for _ in range(vocab_size)]
                target = rng.randrange(vocab_size)
                with self.subTest(vocab_size=vocab_size, target=target):
                    result = compare_sample(scores, target)
                    self.assertEqual(result["summary"]["failed"], 0)
                    # 用乘积归一的另一标量表达式核对概率与损失。
                    odds = [math.exp(s - scores[target]) for s in scores]
                    independent_loss = math.log(math.fsum(odds))
                    self.assertAlmostEqual(result["loss"], independent_loss, places=12)
                    self.assertAlmostEqual(math.fsum(result["autograd_gradient"]), 0, places=12)

    def test_translation_and_permutation(self):
        scores, target = [0.7, -1.3, 2.1, 0.2], 1
        original = compare_sample(scores, target)
        shifted = compare_sample([s + 1000 for s in scores], target)
        permutation = [2, 0, 3, 1]
        permuted = compare_sample([scores[i] for i in permutation], permutation.index(target))
        self.assertEqual(shifted["summary"]["failed"], 0)
        self.assertEqual(permuted["summary"]["failed"], 0)
        for i, g in enumerate(original["autograd_gradient"]):
            self.assertAlmostEqual(g, shifted["autograd_gradient"][i], places=12)
            self.assertAlmostEqual(g, permuted["autograd_gradient"][permutation.index(i)], places=12)

    def test_coupled_matrix_difference_and_preservation(self):
        # 同时有各格平方与跨格乘积，检查逐格扰动/形状和返回顺序。
        base = torch.tensor([[0.2, -0.7], [1.1, 0.4]], dtype=DTYPE, requires_grad=True)
        base.grad = torch.full_like(base, 5)
        original, original_grad = base.detach().clone(), base.grad.clone()
        def loss(v):
            return (v * v).sum() + 3 * v[0, 0] * v[1, 1]
        numeric = central_difference(base, loss)
        expected = torch.tensor([[2*.2 + 3*.4, 2*(-.7)], [2*1.1, 2*.4 + 3*.2]], dtype=DTYPE)
        self.assertTrue(torch.allclose(numeric, expected, atol=1e-9, rtol=0))
        self.assertTrue(torch.equal(base, original))
        self.assertTrue(torch.equal(base.grad, original_grad))

    def test_nonfinite_loss_is_not_silenced(self):
        point = torch.tensor([1.0], dtype=DTYPE)
        with self.assertRaises(FloatingPointError):
            central_difference(point, lambda v: v.sum() * float("nan"))
        with self.assertRaises(FloatingPointError):
            reference([1e308, -1e308], 1)

    def test_required_input_rejections(self):
        checks = Checks()
        records = run.validate_bad_inputs(checks)
        self.assertEqual(len(records["rejections"]), 22)
        self.assertEqual(checks.summary()["failed"], 0)

    def test_close_rejects_shape_nan_and_wrong_numeric(self):
        checks = Checks()
        checks.close("shape", [1, 2], [[1, 2]])
        checks.close("nan", [float("nan")], [0])
        checks.close("wrong", [1], [0])
        self.assertEqual(checks.summary()["failed"], 3)

    def test_unexpected_experiment_exception_is_saved_and_fails(self):
        def broken(checks):
            raise RuntimeError("注入非预期实验错误")
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "failure.json"
            with patch.dict(run.EXPERIMENTS, {"injected": broken}, clear=True):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    code = run.main(["--output", str(output)])
            result = json.loads(output.read_text())
            self.assertEqual(code, 1)
            self.assertEqual(result["summary"]["status"], "failed")
            self.assertIn("注入非预期实验错误", result["experiments"]["injected"]["observations"]["unexpected_error"]["message"])
            self.assertIn("input_validation", result["experiments"])

    def test_numeric_failure_is_saved_and_fails(self):
        def wrong(checks):
            checks.close("错误梯度", [1], [0])
            return {"actual": [1], "expected": [0]}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "wrong.json"
            with patch.dict(run.EXPERIMENTS, {"injected": wrong}, clear=True):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    code = run.main(["--output", str(output)])
            result = json.loads(output.read_text())
            self.assertEqual(code, 1)
            self.assertEqual(result["summary"]["failed_names"], ["injected/错误梯度"])

    def test_cli_subprocess_and_saved_results(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "results.json"
            process = subprocess.run(
                [sys.executable, "-B", "-m", "learning.stage03_gradients.run", "--output", str(output)],
                cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["summary"]["failed"], 0)
            self.assertTrue(all(report["experiments"][key]["summary"]["failed"] == 0 for key in "ABCDEF"))
            self.assertEqual(report["experiments"]["B"]["observations"]["states"]["step_without_backward"]["grad"], None)
            self.assertEqual(report["experiments"]["E"]["observations"]["detached_used"]["outcome"], "expected_error")
            self.assertEqual(len(report["experiments"]["A"]["observations"]["step_size_scan"]["rows"]), 4)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GradientChecks))
    output = Path(__file__).resolve().parent / "output" / "checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "test_methods": result.testsRun, "random_samples": 20,
        "passed": result.wasSuccessful(), "failures": [str(t) for t, _ in result.failures],
        "errors": [str(t) for t, _ in result.errors],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
