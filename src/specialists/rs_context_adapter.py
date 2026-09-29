"""Lightweight, real remote-sensing domain-context layer for VQA prompts.

WHAT THIS IS: a small, switchable "adapter interface" that injects a short
list of real, publicly documented remote-sensing vocabulary terms into the
text prompt sent to the VQA model, when the terms are relevant to the
user's question. It is prompt-time context only.

WHAT THIS IS NOT: this is NOT fine-tuning, NOT a trained adapter (e.g. not
a LoRA/PEFT weight delta), and it does NOT change the model's weights or
claim the base model was ever exposed to this vocabulary during training.
`specialists/vqa_smolvlm.py` still runs the exact same pretrained,
general-purpose `HuggingFaceTB/SmolVLM-256M-Instruct` checkpoint - this
module only changes what text is written into that model's input prompt,
via `build_domain_aware_prompt()`.

PROVENANCE OF `BIGEARTHNET_19_CLASSES` (read this before changing it):
This vocabulary was NOT extracted from any file in this repository and was
NOT produced by processing a local dataset. No `BigEarthNet.txt` or any
BigEarthNet annotation/label file exists anywhere in this repo (confirmed
by a full-repo search) - `data/raw/sample_pair_bigearthnet/` holds only
raw, unlabeled Sentinel-1/2 GeoTIFF band imagery, no class labels. Instead,
these 19 class names are the public BigEarthNet-19 simplified label set's
own documented nomenclature, as defined by the BigEarthNet dataset paper
and mirrored verbatim in torchgeo's open-source `bigearthnet.py` dataset
loader (`class_sets[19]`). They are reproduced here, hardcoded, as a small
(~19-line) cited reference list - not downloaded, not training data, not a
vocabulary this project ever trained on. `docs/rs_adaptation.md` already
documents that this project deliberately chose VRSBench over BigEarthNet
for its (never-executed) fine-tuning research, precisely because
BigEarthNet is multi-label classification data, not VQA/grounding data -
that reasoning is unaffected by this module, which uses the BigEarthNet
taxonomy only as reference vocabulary for prompt context, never as
training data and never as a claim that fine-tuning occurred.

Every place this vocabulary is surfaced (this docstring, the UI badge, the
execution raw-data trace) must describe it as "BigEarthNet-19 land-cover
taxonomy (public dataset nomenclature)" - see `SOURCE` below - and must
NOT say or imply "extracted from BigEarthNet.txt in this repository" or
"extracted from local training data," since neither is true.
"""
from __future__ import annotations

import dataclasses
import re
from typing import Dict, FrozenSet, List, Tuple

# The public BigEarthNet-19 simplified label set, reproduced verbatim from
# the dataset's own documented nomenclature (cross-checked against
# torchgeo's `bigearthnet.py` `class_sets[19]`, which mirrors the original
# BigEarthNet paper). See the module docstring above for the full
# provenance statement - this is NOT data found in this repository.
BIGEARTHNET_19_CLASSES: Tuple[str, ...] = (
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
)

# Honest, single-sentence provenance label - reused verbatim by the UI badge
# and by the execution trace, so the disclosure text can never drift out of
# sync across surfaces. See the module docstring's PROVENANCE section.
SOURCE = "BigEarthNet-19 land-cover taxonomy (public dataset nomenclature)"

# Single, clearly-named switch for the whole feature. Defaults to on since
# real, honestly-sourced vocabulary backs it - flip to False to cleanly
# disable prompt-time domain-context injection everywhere it is wired in
# (`specialists/vqa_smolvlm.py:run()`), without deleting any code.
ADAPTATION_ENABLED = True


@dataclasses.dataclass
class RSContextAdaptation:
    """Small, typed record of what this layer actually did for one query.

    `applied` is True only when `terms` is non-empty AND the terms were
    genuinely injected into the prompt sent to the model - never set True
    just because the feature is enabled, and never forced True to make the
    UI look more active than the real run was.
    """

    source: str
    terms: List[str]
    applied: bool

    def to_dict(self) -> Dict[str, object]:
        return {"source": self.source, "terms": list(self.terms), "applied": self.applied}


# --- pure term-matching (no ML, no embeddings, no external calls) ---------

# Generic words that appear constantly in remote-sensing questions but
# carry no land-cover-specific meaning on their own (e.g. "this area",
# "in the image") - excluded so they don't produce spurious matches against
# BigEarthNet class names that happen to contain a word like "areas".
_STOPWORDS: FrozenSet[str] = frozenset(
    {
        "a", "an", "and", "any", "area", "areas", "are", "as", "at", "be",
        "by", "can", "could", "describe", "do", "does", "for", "from",
        "how", "image", "images", "in", "into", "is", "it", "its", "kind",
        "kinds", "many", "more", "most", "much", "of", "on", "or", "out",
        "photo", "picture", "region", "regions", "scene", "see", "show",
        "shown", "that", "the", "there", "these", "this", "to", "type",
        "types", "what", "where", "which", "with", "you",
    }
)

_WORD_RE = re.compile(r"[A-Za-z]+")


_ES_PLURAL_RE = re.compile(r"(?:s|sh|ch|x|z)es$")


def _normalize(word: str) -> str:
    """Crude, stemming-free de-pluralization - just enough to match
    "waters"/"water", "forests"/"forest", "pastures"/"pasture" without
    pulling in a real stemmer or any ML dependency."""
    w = word.lower()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if _ES_PLURAL_RE.search(w) and len(w) > 4:
        # e.g. "boxes" -> "box", "wetlands" is unaffected (doesn't end in
        # s/sh/ch/x/z + "es"); NOT a generic "any -es -> strip 2" rule,
        # which would wrongly turn "pastures" into "pastur".
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def _content_words(text: str) -> FrozenSet[str]:
    words = _WORD_RE.findall(text or "")
    return frozenset(
        _normalize(w) for w in words if len(w) > 2 and w.lower() not in _STOPWORDS
    )


# Precomputed once at import time (pure, deterministic, no I/O) so
# `retrieve_relevant_terms` doesn't re-tokenize all 19 class names on every
# call.
_CLASS_WORD_SETS: Tuple[FrozenSet[str], ...] = tuple(
    _content_words(name) for name in BIGEARTHNET_19_CLASSES
)


def retrieve_relevant_terms(query: str, max_terms: int = 6) -> List[str]:
    """Score each BigEarthNet-19 class name by case-insensitive, stemming-
    free word overlap against the query's own content words, and return the
    verbatim class names (in `BIGEARTHNET_19_CLASSES` order for ties) with
    the highest overlap, most-relevant first, up to `max_terms`.

    Fully deterministic and inspectable: no model, no embeddings, no
    external calls. If nothing overlaps meaningfully - e.g. a query with no
    land-cover-relevant wording, such as "how many objects are in this
    image?" - this honestly returns an empty list rather than force-filling
    irrelevant terms just to have something to show.
    """
    query_words = _content_words(query)
    if not query_words:
        return []

    scored: List[Tuple[int, int, str]] = []
    for idx, class_words in enumerate(_CLASS_WORD_SETS):
        overlap = len(query_words & class_words)
        if overlap > 0:
            scored.append((overlap, idx, BIGEARTHNET_19_CLASSES[idx]))

    scored.sort(key=lambda t: (-t[0], t[1]))
    return [name for _, _, name in scored[:max_terms]]


def build_domain_aware_prompt(base_query: str, terms: List[str]) -> str:
    """Builds the actual text to send to the model: the original user query
    stays fully intact and primary, with any retrieved terms appended as a
    clearly-delimited, clearly-hedged supporting-context sentence that does
    NOT assert any of the listed terms are true of this specific image.

    Pure and separately unit-testable - no torch, no model, no I/O.
    """
    base = (base_query or "").strip()
    if not terms:
        return base

    term_list = ", ".join(terms)
    return (
        f"{base}\n\n"
        "Reference land-cover vocabulary that may be relevant (not asserted "
        f"to apply to this specific image; source: {SOURCE}): {term_list}."
    )
