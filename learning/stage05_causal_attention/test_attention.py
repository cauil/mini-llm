"""补充验证：非连续输入、独立反向对照、状态保护与检查器的失败出口。"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace

import torch

from . import model as model_module
from .model import AttentionConfig, SingleHeadCausalAttention, causal_attention
from .reference import attention_reference, project_reference
from .check_attention import main


class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_noncontiguous_and_independent_gradients(self):
        """使用新尺寸与逐位置参考的autograd对照，不是只检查.grad存在。"""
        for seed, shape in ((611, (3, 6, 4, 3, 5)), (612, (1, 2, 5, 4, 1))):
            with self.subTest(shape=shape):
                b, t, d, dk, dv = shape
                config = AttentionConfig(d, dk, dv, seed)
                actual_model, reference_model = SingleHeadCausalAttention(config), SingleHeadCausalAttention(config)
                generator = torch.Generator().manual_seed(seed + 1)
                x = torch.randn(b, d, t, dtype=torch.float64, generator=generator).transpose(1, 2).requires_grad_()
                self.assertFalse(x.is_contiguous())
                reference_x = x.detach().clone().requires_grad_()
                h = actual_model(x)
                projections = [project_reference(reference_x, layer.weight)
                               for layer in (reference_model.query, reference_model.key, reference_model.value)]
                ref_h, _ = attention_reference(*projections)
                torch.testing.assert_close(h, ref_h, atol=1e-10, rtol=1e-8)
                r = torch.randn(b, t, dv, dtype=torch.float64, generator=generator)
                (h * r).sum().backward()
                (ref_h * r).sum().backward()
                torch.testing.assert_close(x.grad, reference_x.grad, atol=1e-10, rtol=1e-8)
                for actual, expected in zip(actual_model.parameters(), reference_model.parameters()):
                    torch.testing.assert_close(actual.grad, expected.grad, atol=1e-10, rtol=1e-8)

    def test_initialization_and_forward_preserve_state(self):
        rng = torch.random.get_rng_state().clone()
        model = SingleHeadCausalAttention(AttentionConfig(3, 2, 4, 5))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        x = torch.arange(18, dtype=torch.float64).reshape(2, 3, 3) / 10
        input_before = x.clone()
        parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        training_h = model(x)
        model.eval()
        eval_h = model(x)
        self.assertFalse(model.training)
        torch.testing.assert_close(training_h, eval_h, atol=0, rtol=0)
        for name, p in model.named_parameters():
            self.assertTrue(torch.equal(p, parameters[name]))
            self.assertTrue(torch.equal(p.grad, torch.ones_like(p)))
        self.assertTrue(torch.equal(x, input_before))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_device_mismatch_is_clear(self):
        cpu = torch.ones(1, 2, 3, dtype=torch.float64)
        meta = torch.empty(1, 2, 3, dtype=torch.float64, device="meta")
        with self.assertRaisesRegex(ValueError, "device"):
            causal_attention(cpu, meta, cpu)
        model = SingleHeadCausalAttention(AttentionConfig(3, 2, 2))
        with self.assertRaisesRegex(ValueError, "device"):
            model(meta)

    def test_stable_softmax_on_large_finite_scores(self):
        q = torch.full((1, 3, 1), 1000., dtype=torch.float64)
        k = torch.tensor([[[1.], [1.001], [0.999]]], dtype=torch.float64)
        v = torch.eye(3, dtype=torch.float64).unsqueeze(0)
        h, a = causal_attention(q, k, v)
        ref_h, ref_a = attention_reference(q, k, v)
        torch.testing.assert_close(h, ref_h, atol=1e-10, rtol=1e-8)
        torch.testing.assert_close(a, ref_a, atol=1e-10, rtol=1e-8)
        self.assertTrue(torch.isfinite(h).all())

    def test_checker_exits_nonzero_for_real_forward_fault(self):
        original = model_module.attention_trace

        def remove_mask(q, k, v):
            trace = original(q, k, v)
            wrong_a = trace.scores.softmax(-1)
            return replace(trace, masked_scores=trace.scores, weights=wrong_a, output=wrong_a @ v)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(model_module, "attention_trace", remove_mask), contextlib.redirect_stdout(io.StringIO()):
                code = main(["--output-dir", directory])
            self.assertEqual(code, 1)
            def invalid_constant(value):
                raise AssertionError("非法JSON常量: " + value)
            report = json.loads((Path(directory) / "checks.json").read_text(encoding="utf-8"),
                                parse_constant=invalid_constant)
            self.assertGreater(report["summary"]["failed"], 0)
            names = {i["name"] for i in report["checks"] if not i["passed"]}
            self.assertIn("causal/0/correct", names)

    def test_checker_rejects_fault_that_accidentally_passes(self):
        def actually_correct(model, x, *, fault):
            trace = model.inspect(x)
            return trace.output, trace.weights

        with tempfile.TemporaryDirectory() as directory:
            with patch("learning.stage05_causal_attention.check_attention.broken_attention", actually_correct), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = main(["--output-dir", directory])
            self.assertEqual(code, 1)
            report = json.loads((Path(directory) / "checks.json").read_text(encoding="utf-8"))
            names = {i["name"] for i in report["checks"] if not i["passed"]}
            self.assertIn("causal/0/no_mask/detected", names)
            self.assertIn("causal/1/query_softmax/detected", names)


if __name__ == "__main__":
    unittest.main()
