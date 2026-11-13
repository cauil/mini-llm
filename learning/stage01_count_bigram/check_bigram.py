"""阶段 1 的手算、边界、隔离、生成、采样与 CLI 验收。"""

import copy
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from learning.stage00_char_codec.tokenizer import CharTokenizer
from learning.stage01_count_bigram.bigram import CountBigram, validate_groups
from learning.stage01_count_bigram.run import build_run, run_files


FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]
VOCAB = {"春": 0, "风": 1, "雨": 2}


def documents(*texts):
    return [{"id": str(index), "text": text} for index, text in enumerate(texts)]


class BigramChecks(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((FOLDER / "documents.json").read_text(encoding="utf-8"))
        self.tokenizer = CharTokenizer(VOCAB)
        self.model = CountBigram(self.tokenizer).fit(self.data["train"])

    def test_public_hand_calculation(self):
        self.assertEqual(self.model.counts, [[0, 2, 0], [2, 0, 1], [0, 1, 0]])
        expected = [[1/5, 3/5, 1/5], [1/2, 1/6, 1/3], [1/4, 1/2, 1/4]]
        for actual, reference in zip(self.model.probabilities, expected):
            self.assertAlmostEqual(sum(actual), 1)
            for value, target in zip(actual, reference):
                self.assertGreater(value, 0)
                self.assertAlmostEqual(value, target)
        metrics = self.model.evaluate(self.data["val"])
        self.assertEqual(metrics["pair_count"], 2)
        self.assertAlmostEqual(metrics["nll_sum"], -math.log(0.2) - math.log(0.5))
        self.assertAlmostEqual(metrics["mean_nll"], math.log(10)/2)
        self.assertEqual(metrics["doc_ids"], ["val-1"])
        trace = metrics["documents"][0]["predictions"]
        self.assertEqual([p["target_position"] for p in trace], [1, 2])
        self.assertEqual([p["current_id"] for p in trace], [0, 2])
        self.assertEqual([p["target_id"] for p in trace], [2, 1])
        self.assertEqual([p["probability"] for p in trace], [0.2, 0.5])

    def test_repeated_characters_asymmetry_and_unseen_row(self):
        model = CountBigram(CharTokenizer({"a": 0, "b": 1, "c": 2}))
        model.fit(documents("aaba", "ab"))
        self.assertEqual(model.counts, [[1, 2, 0], [1, 0, 0], [0, 0, 0]])
        self.assertEqual(model.distribution(0), [2/6, 3/6, 1/6])
        self.assertEqual(model.distribution(1), [2/4, 1/4, 1/4])
        self.assertEqual(model.distribution(2), [1/3] * 3)

    def test_whitespace_is_not_removed(self):
        model = CountBigram(CharTokenizer({"a": 0, " ": 1, "\n": 2}))
        model.fit(documents("a a\n"))
        self.assertEqual(model.counts, [[0, 1, 1], [1, 0, 0], [0, 0, 0]])
        self.assertEqual(model.evaluate(documents("a a\n"))["pair_count"], 3)

    def test_document_boundaries_and_empty_groups(self):
        for training in ([], documents("", "春", "风")):
            with self.subTest(training=training):
                model = CountBigram(self.tokenizer).fit(training)
                self.assertEqual(model.counts, [[0]*3 for _ in range(3)])
                self.assertEqual(model.probabilities, [[1/3]*3 for _ in range(3)])
                for validation in ([], documents("", "雨")):
                    result = model.evaluate(validation)
                    self.assertEqual(result["pair_count"], 0)
                    self.assertEqual(result["nll_sum"], 0)
                    self.assertIsNone(result["mean_nll"])
                    self.assertTrue(result["reason"])
        result = build_run({"train": [], "val": []}, self.tokenizer, "春", 0)
        self.assertIsNone(result["evaluation"]["val"]["mean_nll"])

    def test_same_text_different_ids_counts_twice(self):
        self.model.fit(documents("春风", "春风"))
        self.assertEqual(self.model.counts[0][1], 2)

    def test_validation_does_not_change_model(self):
        altered = copy.deepcopy(self.data)
        altered["val"][0]["text"] = "雨雨雨春"
        first = build_run(self.data, self.tokenizer, "春")
        second = build_run(altered, self.tokenizer, "春")
        self.assertEqual(first["model"], second["model"])
        self.assertEqual(first["generations"], second["generations"])
        before = self.model.to_dict()
        self.model.evaluate(altered["val"])
        self.model.generate("雨", 20)
        self.assertEqual(before, self.model.to_dict())
        self.assertEqual(self.tokenizer.vocab, VOCAB)

    def test_refit_replaces_counts_and_failed_fit_keeps_state(self):
        before = self.model.to_dict()
        self.model.fit(self.data["train"])
        self.assertEqual(before, self.model.to_dict())
        with self.assertRaisesRegex(ValueError, "bad.*未知字符.*雪"):
            self.model.fit([{"id": "bad", "text": "雪"}])
        self.assertEqual(before, self.model.to_dict())
        self.model.fit([])
        self.assertEqual(self.model.counts, [[0]*3 for _ in range(3)])

    def test_returned_tables_are_copies(self):
        before = self.model.to_dict()
        self.model.counts[0][0] = 100
        self.model.probabilities[0][0] = 100
        self.model.distribution(0)[0] = 100
        snapshot = self.model.to_dict()
        snapshot["vocab"]["春"] = 100
        self.assertEqual(before, self.model.to_dict())

    def test_weighted_mean_for_unequal_document_lengths(self):
        result = self.model.evaluate(documents("春风", "风春风雨"))
        expected_sum = -math.log(0.6) - math.log(0.5) - math.log(0.6) - math.log(1/3)
        self.assertEqual(result["pair_count"], 4)
        self.assertAlmostEqual(result["nll_sum"], expected_sum)
        self.assertAlmostEqual(result["mean_nll"], expected_sum / 4)
        wrong_mean = sum(doc["mean_nll"] for doc in result["documents"]) / 2
        self.assertNotAlmostEqual(result["mean_nll"], wrong_mean)

    def test_argmax_steps_tie_and_prompt_preservation(self):
        self.assertEqual(self.model.generate("春", 2, "argmax")["text"], "春风春")
        self.assertEqual(self.model.generate("雨", 2, "argmax")["text"], "雨风春")
        for prefix in ("春风", "雨风"):
            self.assertEqual(self.model.generate(prefix, 1, "argmax")["text"], prefix + "春")
        self.assertEqual(self.model.generate("春风", 0)["tokens"], [0, 1])
        uniform = CountBigram(self.tokenizer).fit([])
        self.assertEqual(uniform.generate("雨", 2, "argmax")["text"], "雨春春")

    def test_sample_reproducibility_and_continuous_rng(self):
        # 用独立的区间抽样参考实现，检查完整生成会换行且不逐步重置种子。
        expected_rows = [[1/5, 3/5, 1/5], [1/2, 1/6, 1/3], [1/4, 1/2, 1/4]]
        reference_rng = random.Random(42)
        expected = [0]
        for _ in range(40):
            row = expected_rows[expected[-1]]
            value = reference_rng.random() * sum(row)
            cumulative = 0
            for token_id, probability in enumerate(row):
                cumulative += probability
                if value < cumulative:
                    expected.append(token_id)
                    break
        first = self.model.generate("春", 40, "sample", 42)
        self.assertEqual(first["tokens"], expected)
        self.assertEqual(len(first["text"]), 41)
        self.assertEqual(first, self.model.generate("春", 40, "sample", 42))
        random.seed(123)
        state = random.getstate()
        self.model.generate("风", 10, "sample", 42)
        self.assertEqual(state, random.getstate())

    def test_fixed_row_sampling_frequency(self):
        rng = random.Random(20261112)
        counts = Counter(self.model.sample_next(1, rng) for _ in range(10000))
        for token_id, expected in enumerate([1/2, 1/6, 1/3]):
            self.assertLessEqual(abs(counts[token_id] / 10000 - expected), 0.03)

    def test_invalid_documents_and_cross_split_ids(self):
        bad_inputs = [None, [], {}, {"train": [], "val": [], "test": []},
                      {"train": {}, "val": []}]
        for data in bad_inputs:
            with self.subTest(data=data), self.assertRaises(ValueError):
                validate_groups(data)
        for doc in [None, {}, {"id": "", "text": "春"},
                    {"id": 0, "text": "春"}, {"id": "a", "text": 1}]:
            with self.subTest(doc=doc), self.assertRaises(ValueError):
                self.model.fit([doc])
        duplicate = {"id": "same", "text": "春"}
        for data in ({"train": [duplicate, duplicate], "val": []},
                     {"train": [duplicate], "val": [duplicate]}):
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, "id 重复"):
                build_run(data, self.tokenizer, "春")

    def test_unknown_characters_and_generation_arguments(self):
        before = self.model.to_dict()
        with self.assertRaisesRegex(ValueError, "short.*未知字符.*雪"):
            self.model.evaluate([{"id": "short", "text": "雪"}])
        for split in ("train", "val"):
            data = copy.deepcopy(self.data)
            data[split][0]["text"] = "雪"
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "未知字符"):
                build_run(data, self.tokenizer, "春")
        for prompt in ("", None, "雪春", "春雪"):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                self.model.generate(prompt, 0)
        for length in (-1, True, 1.5, "2"):
            with self.subTest(length=length), self.assertRaises(ValueError):
                self.model.generate("春", length)
        for seed in (True, 1.5, "42", None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                self.model.generate("春", 1, seed=seed)
        with self.assertRaises(ValueError):
            self.model.generate("春", 1, method="invalid")
        for token_id in (-1, 3, True, 1.5):
            with self.subTest(token_id=token_id), self.assertRaises(ValueError):
                self.model.distribution(token_id)
        self.assertEqual(before, self.model.to_dict())

    def test_unfitted_operations_fail(self):
        model = CountBigram(self.tokenizer)
        for operation in (lambda: model.counts, lambda: model.distribution(0),
                          lambda: model.evaluate([]), lambda: model.generate("春", 0)):
            with self.assertRaisesRegex(ValueError, "先调用 fit"):
                operation()

    def test_random_corpora_against_position_reference(self):
        rng = random.Random(7)
        for case in range(100):
            texts = ["".join(rng.choices("春风雨", k=rng.randrange(9)))
                     for _ in range(rng.randrange(6))]
            self.model.fit(documents(*texts))
            # 以字符对 Counter 建独立参考，不调用模型计数或归一化代码。
            reference = Counter((text[i], text[i+1])
                                for text in texts for i in range(len(text)-1))
            for current, current_id in VOCAB.items():
                total = sum(value for (left, _), value in reference.items() if left == current)
                for target, target_id in VOCAB.items():
                    with self.subTest(case=case, pair=(current, target)):
                        count = reference[current, target]
                        self.assertEqual(self.model.counts[current_id][target_id], count)
                        self.assertAlmostEqual(self.model.distribution(current_id)[target_id],
                                               (count+1)/(total+3))

    def test_save_reload_repeat_and_invalid_input_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            first = run_files(FOLDER / "documents.json", FOLDER / "vocab.json", folder, "春")
            saved = folder / "results.json"
            self.assertEqual(json.loads(saved.read_text(encoding="utf-8")), first)
            before = saved.read_bytes()
            second = run_files(FOLDER / "documents.json", FOLDER / "vocab.json", folder, "春")
            self.assertEqual(first, second)
            self.assertEqual(before, saved.read_bytes())
            invalid = folder / "bad.json"
            invalid.write_text('{"train":[],"val":[{"id":"bad","text":"雪"}]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bad.*未知字符"):
                run_files(invalid, FOLDER / "vocab.json", folder, "春")
            self.assertEqual(before, saved.read_bytes())
            invalid.write_text('{"train":[],"val":[]}', encoding="utf-8")
            result = run_files(invalid, FOLDER / "vocab.json", folder, "春", 0)
            self.assertEqual(json.loads(saved.read_text(encoding="utf-8")), result)
            self.assertIsNone(result["evaluation"]["val"]["mean_nll"])
            self.assertIn('"mean_nll": null', saved.read_text(encoding="utf-8"))

    def test_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            command = [sys.executable, "-B", "-m", "learning.stage01_count_bigram.run",
                       "--output-dir", temporary, "--prompt", "雨", "--num-new-tokens", "2"]
            run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
            self.assertIn("argmax: 雨风春", run.stdout)
            result = json.loads((Path(temporary) / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(result["generations"][0]["text"], "雨风春")
            failed = subprocess.run(command + ["--num-new-tokens", "-1"], cwd=ROOT,
                                    capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("新增字符数", failed.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
