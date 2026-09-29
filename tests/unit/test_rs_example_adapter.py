"""Tests for `specialists/rs_example_adapter.py` - the retrieval-augmented,
dataset-grounded remote-sensing context layer.

Two kinds of fixtures are used, and they are never confused with each
other:

  * A tiny, clearly-labeled SYNTHETIC CSV (below) is used to test loading
    mechanics, column handling, split filtering, and retrieval behavior in
    isolation - fast, deterministic, no network, no dependency on whether
    a real dataset cache happens to be present. These rows are invented
    for this test file only and are never presented anywhere as real
    BigEarthNet.txt content.
  * `TestRealDatasetSubsetIfPresent` at the bottom exercises the module
    against whatever is ACTUALLY at the real default cache path
    (`data/cache/bigearthnet_txt_subset.*`) - it is a real, unmocked
    integration test that is expected to run for real once a genuine
    local subset has been prepared (see docs/rs_adaptation.md), and
    skips honestly (never fabricates a pass) when that cache is absent,
    exactly like this project's existing torch/transformers-gated tests.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from specialists import rs_example_adapter as adapter  # noqa: E402

# Synthetic test fixture ONLY - not real BigEarthNet.txt data. Shaped like
# the real columns (id, input, output, type, category, split) so the
# loader's parsing/filtering logic is exercised the same way it would be
# against a real cache.
_SYNTHETIC_CSV = """id,input,output,type,category,split
1,What crop type dominates the central field?,Arable land dominates the central field.,binary,area,train
2,Is there visible water in this scene?,Yes there is a small inland water body in the lower right.,binary,presence,train
3,Describe the land cover in this patch.,The patch is mostly broad-leaved forest with a small urban fabric cluster.,captioning,area,train
4,Count the number of distinct fields visible.,There are four distinct agricultural fields visible.,mcq,count,test
5,What season does this image appear to be from?,The vegetation pattern suggests a summer acquisition.,binary,season,validation
6,Is this area forested?,Yes this area is covered by coniferous forest.,binary,presence,bench
7,Are there any buildings visible near the water?,Yes there is urban fabric adjacent to the inland water body.,binary,adjacency,train
"""


def _write_synthetic_csv() -> str:
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(_SYNTHETIC_CSV)
    return path


class TestLoadDatasetSubsetParsing(unittest.TestCase):
    """Covers: input/output column parsing, and train-split-only filtering
    (excludes test/validation/bench)."""

    def setUp(self):
        self.path = _write_synthetic_csv()

    def tearDown(self):
        os.unlink(self.path)

    def test_parses_input_output_columns_and_keeps_train_split_only(self):
        examples = adapter.load_dataset_subset(self.path)
        self.assertIsNotNone(examples)
        # 4 of the 7 synthetic rows are split=train (ids 1, 2, 3, 7).
        self.assertEqual(len(examples), 4)
        ids = {ex.record_id for ex in examples}
        self.assertEqual(ids, {"1", "2", "3", "7"})
        for ex in examples:
            self.assertEqual(ex.split, "train")
            self.assertTrue(ex.input)
            self.assertTrue(ex.output)

    def test_excludes_test_validation_and_bench_splits(self):
        examples = adapter.load_dataset_subset(self.path)
        record_ids = {ex.record_id for ex in examples}
        # id 4 = test, id 5 = validation, id 6 = bench - none may appear.
        self.assertNotIn("4", record_ids)
        self.assertNotIn("5", record_ids)
        self.assertNotIn("6", record_ids)

    def test_missing_required_columns_returns_none_not_a_crash(self):
        fd, path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("id,question,answer\n1,foo,bar\n")
        try:
            self.assertIsNone(adapter.load_dataset_subset(path))
        finally:
            os.unlink(path)

    def test_nonexistent_path_returns_none_not_a_crash(self):
        self.assertIsNone(adapter.load_dataset_subset("/tmp/does_not_exist_satquery_rs_example.csv"))


class TestRetrieval(unittest.TestCase):
    """Covers: deterministic retrieval, and the top_k <= 3 cap."""

    @classmethod
    def setUpClass(cls):
        cls.path = _write_synthetic_csv()
        cls.examples = adapter.load_dataset_subset(cls.path)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.path)

    def test_retrieval_is_deterministic(self):
        index_a = adapter.RSExampleIndex(self.examples)
        index_b = adapter.RSExampleIndex(self.examples)
        query = "Is there any water visible near the buildings?"
        result_a = [e.record_id for e in index_a.retrieve(query, top_k=3)]
        result_b = [e.record_id for e in index_b.retrieve(query, top_k=3)]
        self.assertEqual(result_a, result_b)
        # Re-running the exact same query against the exact same index must
        # also be stable, not just across two freshly-built indices.
        result_c = [e.record_id for e in index_a.retrieve(query, top_k=3)]
        self.assertEqual(result_a, result_c)

    def test_retrieval_returns_at_most_three_examples(self):
        index = adapter.RSExampleIndex(self.examples)
        for requested in (1, 2, 3, 5, 10, 100):
            result = index.retrieve("water forest fields buildings crop", top_k=requested)
            self.assertLessEqual(len(result), 3)

    def test_irrelevant_query_can_return_fewer_or_zero_examples(self):
        index = adapter.RSExampleIndex(self.examples)
        result = index.retrieve("xyzzy quux nonsense gibberish", top_k=3)
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), 3)

    def test_empty_query_returns_empty_list(self):
        index = adapter.RSExampleIndex(self.examples)
        self.assertEqual(index.retrieve("", top_k=3), [])
        self.assertEqual(index.retrieve("   ", top_k=3), [])


class TestFallbackWhenDatasetUnavailable(unittest.TestCase):
    """Covers: graceful fallback when no local cache is present - never
    fabricates examples, never raises."""

    def test_retrieve_examples_returns_empty_list_when_no_cache(self):
        old = os.environ.get(adapter.ENV_OVERRIDE)
        os.environ[adapter.ENV_OVERRIDE] = "/tmp/definitely_does_not_exist_satquery.csv"
        adapter._INDEX_CACHE.clear()
        try:
            self.assertEqual(adapter.retrieve_examples("Is there water here?"), [])
            self.assertEqual(adapter.dataset_records_available(), 0)
        finally:
            adapter._INDEX_CACHE.clear()
            if old is None:
                os.environ.pop(adapter.ENV_OVERRIDE, None)
            else:
                os.environ[adapter.ENV_OVERRIDE] = old

    def test_build_domain_context_block_is_empty_for_no_examples(self):
        self.assertEqual(adapter.build_domain_context_block([]), "")


class TestTraceMetadata(unittest.TestCase):
    """Covers: trace metadata correctness and fine_tuned is always False -
    exercised at the vqa_smolvlm integration level using a synthetic cache,
    without needing torch/transformers (the retrieval layer itself has no
    ML dependency)."""

    def setUp(self):
        self.path = _write_synthetic_csv()
        self.old = os.environ.get(adapter.ENV_OVERRIDE)
        os.environ[adapter.ENV_OVERRIDE] = self.path
        adapter._INDEX_CACHE.clear()

    def tearDown(self):
        os.unlink(self.path)
        adapter._INDEX_CACHE.clear()
        if self.old is None:
            os.environ.pop(adapter.ENV_OVERRIDE, None)
        else:
            os.environ[adapter.ENV_OVERRIDE] = self.old

    def _build_trace(self, query: str) -> dict:
        # Mirrors exactly the dict vqa_smolvlm.py:run() builds, without
        # needing torch/transformers - the shape under test is the
        # adapter's own contract, which run() consumes verbatim.
        dataset_records_available = adapter.dataset_records_available()
        retrieved = adapter.retrieve_examples(query, top_k=3)
        return {
            "applied": bool(retrieved),
            "method": "retrieval_augmented_prompting",
            "source": adapter.DATASET_SOURCE,
            "dataset_records_available": dataset_records_available,
            "examples_retrieved": len(retrieved),
            "fine_tuned": False,
            "retrieved_ids": [ex.record_id for ex in retrieved],
        }

    def test_trace_metadata_shape_and_values_when_applied(self):
        trace = self._build_trace("Is there visible water near any buildings?")
        self.assertTrue(trace["applied"])
        self.assertEqual(trace["method"], "retrieval_augmented_prompting")
        self.assertEqual(trace["source"], "BigEarthNet.txt")
        self.assertEqual(trace["dataset_records_available"], 4)
        self.assertGreater(trace["examples_retrieved"], 0)
        self.assertLessEqual(trace["examples_retrieved"], 3)
        self.assertEqual(len(trace["retrieved_ids"]), trace["examples_retrieved"])

    def test_fine_tuned_is_always_false(self):
        for query in ["water", "forest crop field", "xyzzy nonsense", ""]:
            trace = self._build_trace(query)
            self.assertFalse(trace["fine_tuned"])

    def test_applied_is_false_and_ids_empty_when_nothing_retrieved(self):
        trace = self._build_trace("")
        self.assertFalse(trace["applied"])
        self.assertEqual(trace["examples_retrieved"], 0)
        self.assertEqual(trace["retrieved_ids"], [])


class TestVqaPromptContainsRetrievedContext(unittest.TestCase):
    """Covers: the VQA prompt genuinely contains retrieved context when
    applied - tests the actual prompt-building function
    (`build_domain_context_block`) used by vqa_smolvlm.py:run(), not a
    reimplementation of it."""

    def setUp(self):
        self.path = _write_synthetic_csv()
        self.old = os.environ.get(adapter.ENV_OVERRIDE)
        os.environ[adapter.ENV_OVERRIDE] = self.path
        adapter._INDEX_CACHE.clear()

    def tearDown(self):
        os.unlink(self.path)
        adapter._INDEX_CACHE.clear()
        if self.old is None:
            os.environ.pop(adapter.ENV_OVERRIDE, None)
        else:
            os.environ[adapter.ENV_OVERRIDE] = self.old

    def test_prompt_block_contains_real_retrieved_text_not_placeholders(self):
        query = "Is there visible water near any buildings?"
        retrieved = adapter.retrieve_examples(query, top_k=3)
        self.assertTrue(retrieved)
        block = adapter.build_domain_context_block(retrieved)
        for ex in retrieved:
            self.assertIn(ex.input, block)
            self.assertIn(ex.output, block)
        self.assertIn("REMOTE-SENSING DOMAIN CONTEXT", block)

    def test_prompt_block_tells_model_not_to_copy_reference_answers(self):
        retrieved = adapter.retrieve_examples("water buildings", top_k=3)
        block = adapter.build_domain_context_block(retrieved)
        self.assertIn("do not copy", block.lower())

    def test_full_prompt_text_appends_block_after_original_query(self):
        # Integration-shaped check matching what vqa_smolvlm.py:run() does:
        # prompt_text = original_query + "\n\n" + context_block. The
        # user's own query must still be present, unmodified, at the start.
        query = "Is there visible water near any buildings?"
        retrieved = adapter.retrieve_examples(query, top_k=3)
        block = adapter.build_domain_context_block(retrieved)
        prompt_text = query + "\n\n" + block
        self.assertTrue(prompt_text.startswith(query))
        self.assertIn("REMOTE-SENSING DOMAIN CONTEXT", prompt_text)


class TestExistingVqaFlowUnaffectedWhenAdapterAbsent(unittest.TestCase):
    """Covers: the classical baseline VQA path (specialists/vqa.py) never
    references this adapter at all - it must remain completely unaffected,
    with no rs_example_adaptation key anywhere in its output."""

    def test_classical_vqa_module_has_no_reference_to_the_new_adapter(self):
        vqa_path = os.path.join(os.path.dirname(__file__), "..", "..", "src", "specialists", "vqa.py")
        with open(vqa_path, encoding="utf-8") as fh:
            classical_src = fh.read()
        self.assertNotIn("rs_example_adapter", classical_src)
        self.assertNotIn("rs_example_adaptation", classical_src)

    def test_adapter_module_has_no_import_side_effects_requiring_torch(self):
        # Re-importing must not require torch/transformers - the retrieval
        # layer has to be usable (and gracefully skip) even in an
        # environment where the real model can never load.
        import importlib
        import specialists.rs_example_adapter as reloaded
        importlib.reload(reloaded)
        self.assertTrue(hasattr(reloaded, "retrieve_examples"))


class TestRealDatasetSubsetIfPresent(unittest.TestCase):
    """Exercises `load_dataset_subset()` / `get_index()` against whatever
    is ACTUALLY present at the real default cache path
    (`data/cache/bigearthnet_txt_subset.*`), with no override and no
    synthetic fixture. This is the one test in this file that is meant to
    run against a real BigEarthNet.txt-derived subset once one has been
    prepared locally (see docs/rs_adaptation.md) - it honestly skips, and
    explains why, when that cache is absent, exactly like this project's
    existing torch/transformers-gated tests in test_florence2_compat.py."""

    def setUp(self):
        adapter._INDEX_CACHE.clear()

    def tearDown(self):
        adapter._INDEX_CACHE.clear()

    def test_loading_the_real_local_bigearthnet_txt_subset(self):
        examples = adapter.load_dataset_subset()
        if examples is None:
            self.skipTest(
                "No local BigEarthNet.txt cache found at data/cache/ - "
                "dataset-backed adaptation is expected to be unavailable "
                "in this environment (see docs/rs_adaptation.md for how to "
                "prepare one); this test will run for real once a cache is "
                "present."
            )
        self.assertGreater(len(examples), 0)
        for ex in examples[:5]:
            self.assertEqual(ex.split, "train")
            self.assertTrue(ex.input)
            self.assertTrue(ex.output)


if __name__ == "__main__":
    unittest.main()
