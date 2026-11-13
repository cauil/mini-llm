"""0C 工程断言；不替代用户的独立理解验收。"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from learning.stage00_char_codec.tokenizer import CharTokenizer
from .prepare import build_windows, prepare_dataset, prepare_files


FOLDER = Path(__file__).resolve().parent


class DatasetChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CharTokenizer.load(FOLDER / "vocab.json")
        self.data = json.loads((FOLDER / "documents.json").read_text(encoding="utf-8"))

    def test_two_hand_calculated_windows(self) -> None:
        windows = build_windows([0, 1, 2, 3, 4, 5], 3)
        self.assertEqual(windows[0], {
            "start": 0, "x": [0, 1, 2], "y": [1, 2, 3],
        })
        self.assertEqual(windows[1], {
            "start": 1, "x": [1, 2, 3], "y": [2, 3, 4],
        })

    def test_public_example_and_last_start(self) -> None:
        result = prepare_dataset(self.data, self.tokenizer, 3)
        self.assertEqual(result["train"], [
            {"doc_id": "A", "start": 0, "x": [0, 1, 2], "y": [1, 2, 3]},
            {"doc_id": "A", "start": 1, "x": [1, 2, 3], "y": [2, 3, 4]},
            {"doc_id": "A", "start": 2, "x": [2, 3, 4], "y": [3, 4, 5]},
        ])
        self.assertEqual(result["val"], [
            {"doc_id": "B", "start": 0, "x": [4, 5, 0], "y": [5, 0, 1]},
        ])

    def test_length_boundaries_and_T_one(self) -> None:
        for T in (1, 3):
            for N in (0, T - 1, T, T + 1, T + 3):
                with self.subTest(T=T, N=N):
                    tokens = list(range(N))
                    windows = build_windows(tokens, T)
                    self.assertEqual(len(windows), max(0, N - T))
                    self.assertEqual([w["start"] for w in windows],
                                     list(range(max(0, N - T))))
        self.assertEqual(build_windows([2, 0, 1], 1), [
            {"start": 0, "x": [2], "y": [0]},
            {"start": 1, "x": [0], "y": [1]},
        ])

    def test_source_trace_whitespace_repetition_and_order(self) -> None:
        tokenizer = CharTokenizer({"甲": 0, "乙": 1, " ": 2, "\n": 3})
        data = {
            "train": [
                {"id": "Z", "text": "甲甲 \n乙甲"},
                {"id": "A", "text": "乙\n 甲乙"},
            ],
            "val": [{"id": "V", "text": "\n甲 甲乙"}],
        }
        result = prepare_dataset(data, tokenizer, 2)
        for split in ("train", "val"):
            expected_order = []
            by_id = {doc["id"]: doc["text"] for doc in data[split]}
            for doc in data[split]:
                expected_order.extend((doc["id"], start)
                                      for start in range(len(doc["text"]) - 2))
            self.assertEqual(
                [(sample["doc_id"], sample["start"]) for sample in result[split]],
                expected_order,
            )
            for sample in result[split]:
                text = by_id[sample["doc_id"]]
                start = sample["start"]
                self.assertEqual(len(sample["x"]), 2)
                self.assertEqual(len(sample["y"]), 2)
                self.assertEqual(tokenizer.decode(sample["x"]), text[start:start + 2])
                self.assertEqual(tokenizer.decode(sample["y"]), text[start + 1:start + 3])
        self.assertEqual(result["train"][0], {
            "doc_id": "Z", "start": 0, "x": [0, 0], "y": [0, 2],
        })

    def test_short_documents_never_join(self) -> None:
        for second_split in ("train", "val"):
            with self.subTest(second_split=second_split):
                data = {"train": [{"id": "A", "text": "春风"}], "val": []}
                data[second_split].append({"id": "B", "text": "山河"})
                result = prepare_dataset(data, self.tokenizer, 3)
                self.assertEqual(result["train"], [])
                self.assertEqual(result["val"], [])
                self.assertEqual([doc["sample_count"]
                                  for doc in result["metadata"]["documents"]], [0, 0])

    def test_empty_groups_and_document(self) -> None:
        for data in (
            {"train": [], "val": []},
            {"train": [{"id": "empty", "text": ""}], "val": []},
            {"train": [], "val": [{"id": "empty", "text": ""}]},
        ):
            with self.subTest(data=data):
                result = prepare_dataset(data, self.tokenizer, 3)
                self.assertEqual(result["train"], [])
                self.assertEqual(result["val"], [])
                for doc in result["metadata"]["documents"]:
                    self.assertEqual(doc["token_count"], 0)
                    self.assertEqual(doc["sample_count"], 0)

    def test_invalid_context_lengths(self) -> None:
        for T in (True, False, 0, -1, 1.5, "3", None):
            with self.subTest(T=T):
                with self.assertRaises(ValueError):
                    build_windows([], T)
                with self.assertRaises(ValueError):
                    prepare_dataset(self.data, self.tokenizer, T)
                with TemporaryDirectory() as folder:
                    output = Path(folder) / "output"
                    with self.assertRaises(ValueError):
                        prepare_files(FOLDER / "documents.json", FOLDER / "vocab.json",
                                      T, output)
                    self.assertFalse(output.exists())

    def test_invalid_token_lists(self) -> None:
        for tokens in ([True], [-1], [1.0], ["0"], (0, 1), None):
            with self.subTest(tokens=tokens):
                with self.assertRaises(ValueError):
                    build_windows(tokens, 1)

    def test_duplicate_ids_within_and_across_splits(self) -> None:
        for split in ("train", "val"):
            for second_split in ("train", "val"):
                with self.subTest(split=split, second_split=second_split):
                    data = {"train": [], "val": []}
                    data[split].append({"id": "same", "text": "春风"})
                    data[second_split].append({"id": "same", "text": "山河"})
                    with self.assertRaisesRegex(ValueError, "same"):
                        prepare_dataset(data, self.tokenizer, 3)

    def test_invalid_document_schema(self) -> None:
        invalid = [
            None, [], {}, {"train": []}, {"train": {}, "val": []},
            {"train": [], "val": None},
        ]
        for doc in (None, {}, {"id": "", "text": ""}, {"id": 1, "text": ""},
                    {"id": "A"}, {"id": "A", "text": None},
                    {"id": "A", "text": 12}):
            for split in ("train", "val"):
                data = {"train": [], "val": []}
                data[split].append(doc)
                invalid.append(data)
        for data in invalid:
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    prepare_dataset(data, self.tokenizer, 3)

    def test_unknown_char_in_either_split_even_if_short(self) -> None:
        original_vocab = self.tokenizer.vocab
        for split in ("train", "val"):
            for text in ("雪", "春风吹过雪"):
                with self.subTest(split=split, text=text):
                    data = {"train": [], "val": []}
                    data[split].append({"id": "unknown-doc", "text": text})
                    with self.assertRaises(KeyError) as raised:
                        prepare_dataset(data, self.tokenizer, 3)
                    message = str(raised.exception)
                    self.assertIn("unknown-doc", message)
                    self.assertIn("雪", message)
                    self.assertIn(split, message)
                    self.assertEqual(self.tokenizer.vocab, original_vocab)

    def test_same_content_different_ids_allowed(self) -> None:
        data = {
            "train": [{"id": "A", "text": "春风吹过"}],
            "val": [{"id": "B", "text": "春风吹过"}],
        }
        result = prepare_dataset(data, self.tokenizer, 3)
        self.assertEqual([s["doc_id"] for s in result["train"]], ["A"])
        self.assertEqual([s["doc_id"] for s in result["val"]], ["B"])

    def test_metadata_persistence_overwrite_and_repeatability(self) -> None:
        self.data["train"].append({"id": "empty", "text": ""})
        with TemporaryDirectory() as folder:
            root = Path(folder)
            documents = root / "documents.json"
            documents.write_text(json.dumps(self.data), encoding="utf-8")
            output = root / "nested" / "output"
            result = prepare_files(documents, FOLDER / "vocab.json", 3, output)
            expected = {
                "train_samples.json": result["train"],
                "val_samples.json": result["val"],
                "metadata.json": result["metadata"],
            }
            before = {}
            for filename, content in expected.items():
                path = output / filename
                self.assertEqual(json.loads(path.read_text(encoding="utf-8")), content)
                before[filename] = path.read_bytes()
            repeated = prepare_files(documents, FOLDER / "vocab.json", 3, output)
            self.assertEqual(repeated, result)
            for filename in expected:
                self.assertEqual((output / filename).read_bytes(), before[filename])
            metadata = result["metadata"]
            self.assertEqual(metadata["T"], 3)
            self.assertEqual(metadata["stride"], 1)
            self.assertEqual(metadata["vocab"], self.tokenizer.vocab)
            self.assertEqual(metadata["train_doc_ids"], ["A", "empty"])
            self.assertEqual(metadata["val_doc_ids"], ["B"])
            self.assertEqual(metadata["documents"], [
                {"doc_id": "A", "split": "train", "token_count": 6, "sample_count": 3},
                {"doc_id": "empty", "split": "train", "token_count": 0, "sample_count": 0},
                {"doc_id": "B", "split": "val", "token_count": 4, "sample_count": 1},
            ])
            for split in ("train", "val"):
                for doc in metadata["documents"]:
                    if doc["split"] == split:
                        self.assertEqual(
                            sum(s["doc_id"] == doc["doc_id"] for s in result[split]),
                            doc["sample_count"],
                        )
            documents.write_text('{"train": [], "val": []}', encoding="utf-8")
            prepare_files(documents, FOLDER / "vocab.json", 3, output)
            for filename in ("train_samples.json", "val_samples.json"):
                self.assertEqual(json.loads((output / filename).read_text()), [])

    def test_invalid_input_does_not_overwrite_existing_output(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "output"
            prepare_files(FOLDER / "documents.json", FOLDER / "vocab.json", 3, output)
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            self.data["val"][0]["text"] = "雪"
            documents = root / "invalid.json"
            documents.write_text(json.dumps(self.data), encoding="utf-8")
            with self.assertRaises(KeyError):
                prepare_files(documents, FOLDER / "vocab.json", 3, output)
            self.assertEqual(
                {path.name: path.read_bytes() for path in output.iterdir()}, before
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
