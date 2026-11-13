"""阶段 2 验收：独立数学参考、状态检查、真实训练与 CLI。"""

import io
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import torch
from torch.nn import functional as F

from learning.stage00_char_codec.tokenizer import CharTokenizer
from .data import prepare_data
from .model import TrainableBigram, generate, make_optimizer
from .run import BASELINE_DIR, OUTPUT_DIR, STAGE_DIR, frequency_experiment, save_results
from .training import evaluate, train


VOCAB = {"春": 0, "风": 1, "雨": 2}
PROJECT_ROOT = STAGE_DIR.parents[1]


def prepare(train_texts, val_texts=(), vocab=None):
    raw = {split: [{"id": f"{split}-{i}", "text": text} for i, text in enumerate(texts)]
           for split, texts in (("train", train_texts), ("val", val_texts))}
    return prepare_data(CharTokenizer(VOCAB if vocab is None else vocab), raw)


def reference(row, target):
    """独立 Python 数学参考，不调用被测模型或 PyTorch softmax。"""
    shifted = [value - max(row) for value in row]
    weights = [math.exp(value) for value in shifted]
    total = math.fsum(weights)
    return [value / total for value in weights], math.log(total) - shifted[target]


class BigramChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_public_samples_and_document_boundaries(self):
        tokenizer = CharTokenizer.load(BASELINE_DIR / "vocab.json")
        raw = json.loads((BASELINE_DIR / "documents.json").read_text())
        data = prepare_data(tokenizer, raw)
        self.assertEqual(data["train"].x.tolist(), [0, 1, 0, 1, 2, 1])
        self.assertEqual(data["train"].y.tolist(), [1, 0, 1, 2, 1, 0])
        self.assertEqual(data["val"].x.tolist(), [0, 2])
        self.assertEqual(data["val"].y.tolist(), [2, 1])
        self.assertEqual(data["train"].samples[2]["doc_id"], "train-2")
        self.assertEqual(data["train"].samples[2]["input_position"], 0)
        boundary = prepare(["", "春", "风", "春风", "春风"])
        self.assertEqual(boundary["train"].x.tolist(), [0, 0])
        self.assertEqual(boundary["train"].y.tolist(), [1, 1])
        self.assertEqual(boundary["train"].x.dtype, torch.long)

    def test_invalid_documents_and_vocab(self):
        tokenizer = CharTokenizer(VOCAB)
        for raw in [
            {"train": [{"id": "a", "text": "春"}], "val": [{"id": "a", "text": "风"}]},
            {"train": [{"id": "", "text": "春"}], "val": []},
            {"train": [{"id": "a", "text": 1}], "val": []},
            {"train": [], "val": [], "test": []},
        ]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                prepare_data(tokenizer, raw)
        with self.assertRaisesRegex(ValueError, "bad-doc"):
            prepare_data(tokenizer, {"train": [], "val": [{"id": "bad-doc", "text": "雪"}]})
        for vocab in ({}, {"春": 1}, {"春": 0, "风": 0}):
            with self.subTest(vocab=vocab), self.assertRaises(ValueError):
                CharTokenizer(vocab)

    def test_shapes_registration_and_zero_loss(self):
        for size in (1, 2, 3, 5):
            with self.subTest(V=size):
                model = TrainableBigram(size)
                x = torch.tensor([size - 1, 0, size - 1, 0], dtype=torch.long)
                logits = model(x)
                self.assertEqual(list(logits.shape), [4, size])
                self.assertEqual(model.parameter_info()["total_trainable"], size * size)
                self.assertEqual(list(dict(model.named_parameters())), ["score_table"])
                optimizer = make_optimizer(model, 0.3)
                self.assertIs(optimizer.param_groups[0]["params"][0], model.score_table)
                self.assertAlmostEqual(F.cross_entropy(logits, x).item(), math.log(size), places=12)
                torch.testing.assert_close(torch.softmax(logits, -1).sum(-1), torch.ones(4, dtype=torch.float64))
                self.assertEqual(tuple(model(torch.tensor([], dtype=torch.long)).shape), (0, size))

    def test_input_and_configuration_errors(self):
        model = TrainableBigram(3)
        for x in (torch.tensor([-1]), torch.tensor([3]), torch.tensor([[0]]),
                  torch.tensor([0.0]), torch.tensor([True]), [0]):
            with self.subTest(x=x), self.assertRaises(ValueError):
                model(x)
        for token in (-1, 3, True, 0.5):
            with self.subTest(token=token), self.assertRaises(ValueError):
                model.distribution(token)
        for size in (0, -1, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                TrainableBigram(size)
        for rate in (0, -0.3, True, float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                make_optimizer(model, rate)
        data = prepare(["春风"])
        for kwargs in ({"steps": -1}, {"steps": True}, {"log_every": 0}, {"log_every": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                train(model, data["train"], data["val"], **kwargs)
        data["train"].y[0] = 3
        with self.assertRaisesRegex(ValueError, "目标.*越界"):
            evaluate(model, data["train"])

    def test_requires_grad_is_not_registration(self):
        model = TrainableBigram(3)
        del model.score_table
        model.score_table = torch.zeros(3, 3, requires_grad=True)
        with self.assertRaisesRegex(ValueError, "注册"):
            make_optimizer(model, 0.3)

    def test_non_symmetric_reference_and_repeated_input(self):
        rows = [[-0.7, 1.4, 0.3], [2.2, -1.1, 0.6], [0.8, 0.1, -1.3]]
        model = TrainableBigram(3)
        with torch.no_grad():
            model.score_table.copy_(torch.tensor(rows))
        # 用模型中实际的浮点值作为独立数学参考输入，避免构造时精度差异。
        rows = model.snapshot()
        x, y = torch.tensor([2, 0, 2, 1]), torch.tensor([1, 2, 0, 1])
        logits = model(x)
        losses = F.cross_entropy(logits, y, reduction="none").tolist()
        for index, (current, target) in enumerate(zip(x.tolist(), y.tolist())):
            expected_probs, expected_loss = reference(rows[current], target)
            for actual, expected in zip(torch.softmax(logits[index], -1).tolist(), expected_probs):
                self.assertAlmostEqual(actual, expected, places=12)
            self.assertAlmostEqual(losses[index], expected_loss, places=12)
        torch.testing.assert_close(logits[0], logits[2])
        self.assertNotEqual(losses[0], losses[2])

    def test_shift_invariance_and_large_logits(self):
        rows = torch.tensor([[1000, 1002, 1001], [-3, 2, 1]], dtype=torch.float64)
        shifted = rows + torch.tensor([[7000], [-1234]], dtype=torch.float64)
        targets = torch.tensor([2, 0])
        torch.testing.assert_close(torch.softmax(rows, -1), torch.softmax(shifted, -1))
        torch.testing.assert_close(F.cross_entropy(rows, targets), F.cross_entropy(shifted, targets))
        expected = sum(reference(row, target)[1]
                       for row, target in zip(rows.tolist(), targets.tolist())) / 2
        self.assertAlmostEqual(F.cross_entropy(rows, targets).item(), expected, places=12)

    def test_underflow_probability_still_finite_nll(self):
        model = TrainableBigram(2)
        with torch.no_grad():
            model.score_table[0] = torch.tensor([0., -1000.])
        data = prepare(["甲乙"], vocab={"甲": 0, "乙": 1})
        result = evaluate(model, data["train"])
        prediction = result["documents"][0]["predictions"][0]
        self.assertEqual(prediction["probability"], 0.0)
        self.assertEqual(prediction["nll"], 1000.0)
        self.assertTrue(math.isfinite(result["mean_nll"]))

    def test_backward_step_and_zero_grad(self):
        model = TrainableBigram(3)
        optimizer = make_optimizer(model, 0.3)
        before = model.score_table.detach().clone()
        F.cross_entropy(model(torch.tensor([0])), torch.tensor([1])).backward()
        torch.testing.assert_close(model.score_table, before)
        expected_grad = torch.tensor([[1/3, -2/3, 1/3], [0, 0, 0], [0, 0, 0]], dtype=torch.float64)
        torch.testing.assert_close(model.score_table.grad, expected_grad)
        optimizer.step()
        expected = torch.tensor([[-0.1, 0.2, -0.1], [0, 0, 0], [0, 0, 0]], dtype=torch.float64)
        torch.testing.assert_close(model.score_table, expected)
        optimizer.zero_grad(set_to_none=True)
        self.assertIsNone(model.score_table.grad)
        torch.testing.assert_close(model.score_table, expected)

    def test_shared_rows_and_stale_gradient_cleared(self):
        data = prepare(["春风", "春雨", "雨春"])
        model = TrainableBigram(3)
        model.score_table.grad = torch.full_like(model.score_table, 100)
        report = train(model, data["train"], data["val"], steps=1)
        expected = torch.tensor([[2/9, -1/9, -1/9], [0, 0, 0], [-2/9, 1/9, 1/9]], dtype=torch.float64)
        torch.testing.assert_close(model.score_table.grad, expected)
        torch.testing.assert_close(model.score_table, -0.3 * expected)
        self.assertEqual(report["first_step"]["before"], report["first_step"]["after_backward"])

    def test_evaluate_preserves_parameters_and_gradients(self):
        data = prepare(["春风雨春"], ["雨春风"])
        model = TrainableBigram(3)
        F.cross_entropy(model(data["train"].x), data["train"].y).backward()
        params = model.score_table.detach().clone()
        grad = model.score_table.grad
        grad_copy = grad.clone()
        evaluate(model, data["val"])
        torch.testing.assert_close(model.score_table, params, rtol=0, atol=0)
        self.assertIs(model.score_table.grad, grad)
        torch.testing.assert_close(grad, grad_copy, rtol=0, atol=0)
        fresh = TrainableBigram(3)
        evaluate(fresh, data["val"])
        self.assertIsNone(fresh.score_table.grad)

    def test_validation_does_not_change_training_trajectory(self):
        first = prepare(["春风春", "雨风春"], ["春雨风"])
        second = prepare(["春风春", "雨风春"], ["雨雨风雨春雨"])
        reports = []
        for data in (first, second):
            model = TrainableBigram(3)
            reports.append(train(model, data["train"], data["val"], steps=9, log_every=1))
        for a, b in zip(reports[0]["history"], reports[1]["history"]):
            self.assertEqual(a["logits"], b["logits"])
            self.assertEqual(a["train"], b["train"])
        self.assertNotEqual(reports[0]["history"][-1]["val"], reports[1]["history"][-1]["val"])

    def test_position_weighted_average(self):
        data = prepare(["春风", "雨春风雨春"])
        model = TrainableBigram(3)
        with torch.no_grad():
            model.score_table.copy_(torch.tensor([[0., 2., -1.], [1., -2., 0.], [2., 1., -1.]]))
        rows = model.snapshot()
        expected_losses = [reference(rows[current], target)[1]
                           for current, target in ((0, 1), (2, 0), (0, 1), (1, 2), (2, 0))]
        result = evaluate(model, data["train"])
        self.assertEqual(result["pair_count"], 5)
        self.assertAlmostEqual(result["nll_sum"], math.fsum(expected_losses), places=12)
        self.assertAlmostEqual(result["mean_nll"], math.fsum(expected_losses) / 5, places=12)
        wrong = sum(d["mean_nll"] for d in result["documents"]) / 2
        self.assertNotAlmostEqual(result["mean_nll"], wrong, places=6)

    def test_empty_training_validation_and_single_token_vocab(self):
        empty = prepare(["", "春", "风"])
        with self.assertRaisesRegex(ValueError, "没有相邻对"):
            train(TrainableBigram(3), empty["train"], empty["val"], steps=0)
        result = evaluate(TrainableBigram(3), empty["train"])
        self.assertEqual(result["pair_count"], 0)
        self.assertEqual(result["nll_sum"], 0)
        self.assertIsNone(result["mean_nll"])
        self.assertTrue(result["reason"])
        data = prepare(["甲甲甲"], vocab={"甲": 0})
        model = TrainableBigram(1)
        report = train(model, data["train"], data["val"], steps=3)
        self.assertEqual(model.snapshot(), [[0.0]])
        self.assertEqual(report["history"][-1]["train"]["mean_nll"], 0)
        self.assertIsNone(report["history"][-1]["val"]["mean_nll"])

    def test_steps_zero_and_log_semantics(self):
        data = prepare(["春风春"])
        model = TrainableBigram(3)
        report = train(model, data["train"], data["val"], steps=0)
        self.assertEqual([item["step"] for item in report["history"]], [0])
        self.assertIsNone(report["first_step"])
        report = train(model, data["train"], data["val"], steps=3, log_every=2)
        self.assertEqual([item["step"] for item in report["history"]], [0, 2, 3])
        for item in report["history"]:
            rows = item["logits"]
            expected = sum(reference(rows[x], y)[1] for x, y in ((0, 1), (1, 0))) / 2
            self.assertAlmostEqual(item["train"]["mean_nll"], expected, places=12)
        self.assertNotEqual(report["first_step"]["after_step"], model.snapshot())

    def test_nonfinite_failure_has_step_and_config(self):
        data = prepare(["春风"])
        model = TrainableBigram(3)
        with torch.no_grad():
            model.score_table[0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "step=0.*lr=0.3"):
            train(model, data["train"], data["val"])
        model = TrainableBigram(3, dtype=torch.float32)
        with self.assertRaisesRegex(FloatingPointError, "step=1.*lr=1e\\+308"):
            train(model, data["train"], data["val"], learning_rate=1e308, steps=1)

    def test_public_training_and_frozen_snapshots(self):
        tokenizer = CharTokenizer.load(BASELINE_DIR / "vocab.json")
        raw = json.loads((BASELINE_DIR / "documents.json").read_text())
        data = prepare_data(tokenizer, raw)
        model = TrainableBigram(3)
        report = train(model, data["train"], data["val"])
        initial, final = report["history"][0], report["history"][-1]
        for split in ("train", "val"):
            self.assertAlmostEqual(initial[split]["mean_nll"], math.log(3), places=12)
        self.assertLess(final["train"]["mean_nll"], initial["train"]["mean_nll"])
        self.assertEqual(final["step"], 300)
        self.assertEqual(final["logits"], model.snapshot())
        self.assertTrue(report["optimizer_manages_score_table"])
        first = report["first_step"]
        for row in range(3):
            for col in range(3):
                self.assertAlmostEqual(first["after_step"][row][col],
                                       first["before"][row][col] - 0.3 * first["gradient"][row][col])
        saved = json.dumps(report)
        with torch.no_grad():
            model.score_table.add_(100)
        self.assertEqual(json.dumps(report), saved)

    def test_full_coverage_frequency_convergence(self):
        result = frequency_experiment()
        self.assertEqual(result["counts"], [[3, 1], [1, 3]])
        self.assertEqual(result["target_unsmoothed_frequencies"], [[0.75, 0.25], [0.25, 0.75]])
        self.assertLessEqual(result["max_absolute_error"], 0.02)
        self.assertTrue(result["passed"])
        self.assertEqual(result["history"][0]["logits"], [[0.0, 0.0], [0.0, 0.0]])

    def test_generation_greedy_ties_lengths_and_sampling(self):
        tokenizer = CharTokenizer(VOCAB)
        model = TrainableBigram(3)
        with torch.no_grad():
            model.score_table.copy_(torch.tensor([[0., 0., 2.], [0., 0., 0.], [0., 3., 0.]]))
        self.assertEqual(generate(model, tokenizer, "春", 3, "argmax")["text"], "春雨风春")
        self.assertEqual(generate(model, tokenizer, "春风", 0)["text"], "春风")
        before = model.snapshot()
        state = random.getstate()
        sampled = generate(model, tokenizer, "春风", 30, "sample", 17)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(sampled, generate(model, tokenizer, "春风", 30, "sample", 17))
        self.assertEqual(len(sampled["tokens"]), 32)
        # 独立模拟连续 RNG，防止实现每步重新播种或总是查询初始行。
        rng = random.Random(17)
        expected = [0, 1]
        rows = model.snapshot()
        for _ in range(30):
            probs, _ = reference(rows[expected[-1]], 0)
            expected.append(rng.choices(range(3), weights=probs, k=1)[0])
        self.assertEqual(sampled["tokens"], expected)
        self.assertEqual(model.snapshot(), before)
        for kwargs in ({"prompt": ""}, {"prompt": "雪春"}, {"num_new_tokens": True},
                       {"num_new_tokens": -1}, {"method": "bad"}, {"seed": True}):
            args = {"prompt": "春", "num_new_tokens": 1, **kwargs}
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                generate(model, tokenizer, **args)

    def test_json_roundtrip_output_scope_and_nonfinite_rejection(self):
        with tempfile.TemporaryDirectory(dir=STAGE_DIR) as temp:
            path = Path(temp) / "results.json"
            original = {"logits": [[0., 2.]], "mean_nll": None}
            save_results(original, path)
            self.assertEqual(json.loads(path.read_text()), original)
            with self.assertRaises(ValueError):
                save_results({"loss": float("nan")}, path)
            self.assertEqual(json.loads(path.read_text()), original)
        with self.assertRaises(ValueError):
            save_results({}, BASELINE_DIR / "output" / "results.json")

    def test_cli_saved_results_baseline_and_failure(self):
        with tempfile.TemporaryDirectory(dir=STAGE_DIR) as temp:
            base = [sys.executable, "-B", "-m", "learning.stage02_trainable_bigram.run",
                    "--output-dir", temp]
            completed = subprocess.run(base + ["--steps", "2", "--log-every", "2"],
                                       cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads((Path(temp) / "results.json").read_text())
            self.assertEqual(result["training"]["history"][-1]["step"], 2)
            self.assertEqual(result["model"]["logits"], result["training"]["history"][-1]["logits"])
            self.assertEqual(result["comparison"]["final_trainable"]["val"]["pair_count"], 2)
            self.assertAlmostEqual(result["comparison"]["count_add_one"]["val"]["mean_nll"],
                                   -(math.log(0.2) + math.log(0.5)) / 2)
            self.assertTrue(result["frequency_experiment"]["passed"])
            failed = subprocess.run(base + ["--steps", "-1"], cwd=PROJECT_ROOT,
                                    text=True, capture_output=True, timeout=60)
            self.assertNotEqual(failed.returncode, 0)
            error = json.loads((Path(temp) / "error.json").read_text())
            self.assertEqual(error["status"], "failed")
            self.assertIn("steps", error["error"])
            failed = subprocess.run(base + ["--learning-rate", "nan"], cwd=PROJECT_ROOT,
                                    text=True, capture_output=True, timeout=60)
            self.assertNotEqual(failed.returncode, 0)
            error = json.loads((Path(temp) / "error.json").read_text())
            self.assertEqual(error["config"]["learning_rate"], "nan")
            self.assertIn("学习率", error["error"])


def main():
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BigramChecks)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    transcript = stream.getvalue()
    print(transcript, end="")
    report = {"passed": result.wasSuccessful(), "tests_run": result.testsRun,
              "failures": [{"test": str(test), "traceback": trace} for test, trace in result.failures],
              "errors": [{"test": str(test), "traceback": trace} for test, trace in result.errors],
              "skipped": result.skipped, "python": sys.version,
              "python_executable": sys.executable, "pytorch": str(torch.__version__),
              "command": "python -B -m learning.stage02_trainable_bigram.check_bigram",
              "transcript": transcript}
    save_results(report, OUTPUT_DIR / "checks.json")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
