"""Focused, fast, torch-free unit tests for
`specialists/rs_context_adapter.py`'s pure functions. No mocking needed -
`retrieve_relevant_terms` and `build_domain_aware_prompt` are deterministic
and have no model or I/O dependency."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from specialists import rs_context_adapter as adapter  # noqa: E402


class TestRetrieveRelevantTerms(unittest.TestCase):
    def test_matches_several_real_terms_for_a_relevant_query(self):
        query = "How much of this region is forest, water, or pasture?"
        terms = adapter.retrieve_relevant_terms(query)

        self.assertTrue(len(terms) >= 3, terms)
        # Every returned term must be a real, verbatim BigEarthNet-19 class
        # name - never invented text.
        for t in terms:
            self.assertIn(t, adapter.BIGEARTHNET_19_CLASSES)
        self.assertIn("Pastures", terms)
        self.assertTrue(any("forest" in t.lower() for t in terms))
        self.assertTrue(any("water" in t.lower() for t in terms))

    def test_no_relevant_wording_returns_empty_list(self):
        query = "How many objects are in this image?"
        terms = adapter.retrieve_relevant_terms(query)
        self.assertEqual(terms, [])

    def test_empty_query_returns_empty_list(self):
        self.assertEqual(adapter.retrieve_relevant_terms(""), [])
        self.assertEqual(adapter.retrieve_relevant_terms("   "), [])

    def test_max_terms_is_respected(self):
        # A query built from many class-name words should still be capped.
        query = "urban forest water pasture wetland grassland crops woodland beach"
        terms = adapter.retrieve_relevant_terms(query, max_terms=3)
        self.assertLessEqual(len(terms), 3)

    def test_case_insensitive_matching(self):
        terms_lower = adapter.retrieve_relevant_terms("tell me about the WATER and FOREST here")
        self.assertTrue(any("water" in t.lower() for t in terms_lower))
        self.assertTrue(any("forest" in t.lower() for t in terms_lower))


class TestBuildDomainAwarePrompt(unittest.TestCase):
    def test_prompt_always_contains_the_original_query_text(self):
        query = "What is the dominant land cover in this scene?"
        terms = ["Broad-leaved forest", "Inland waters"]
        prompt = adapter.build_domain_aware_prompt(query, terms)
        self.assertIn(query, prompt)

    def test_no_terms_returns_query_unchanged(self):
        query = "How many objects are in this image?"
        prompt = adapter.build_domain_aware_prompt(query, [])
        self.assertEqual(prompt, query)

    def test_prompt_does_not_assert_terms_are_true_of_the_image(self):
        query = "Describe this scene."
        terms = ["Pastures", "Mixed forest"]
        prompt = adapter.build_domain_aware_prompt(query, terms)
        self.assertIn("not asserted", prompt.lower())
        for t in terms:
            self.assertIn(t, prompt)

    def test_prompt_cites_the_honest_source(self):
        prompt = adapter.build_domain_aware_prompt("q", ["Pastures"])
        self.assertIn(adapter.SOURCE, prompt)


class TestRSContextAdaptationRecord(unittest.TestCase):
    def test_to_dict_round_trip(self):
        record = adapter.RSContextAdaptation(source=adapter.SOURCE, terms=["Pastures"], applied=True)
        d = record.to_dict()
        self.assertEqual(d["source"], adapter.SOURCE)
        self.assertEqual(d["terms"], ["Pastures"])
        self.assertTrue(d["applied"])


if __name__ == "__main__":
    unittest.main()
