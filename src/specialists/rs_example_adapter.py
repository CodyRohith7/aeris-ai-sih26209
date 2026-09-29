"""Retrieval-augmented remote-sensing example adapter (dataset-grounded).

WHAT THIS IS: a small, deterministic TF-IDF retrieval layer over a *local
cache* of real BigEarthNet.txt text records (question/instruction +
reference-answer pairs). Given a user's VQA query, it retrieves the 2-3
most textually similar real records and hands them to
`specialists/vqa_smolvlm.py` as domain-context examples appended to the
prompt - never as facts about the uploaded image, never as text the model
is told to copy.

WHAT THIS IS NOT: this is NOT fine-tuning. It does not touch SmolVLM's
weights, it does not train anything, and `fine_tuned` is hardcoded False
everywhere this module reports its own status. See the module docstring in
`specialists/rs_context_adapter.py` for the (separate, taxonomy-only)
prompt-time adapter this module complements - that one injects BigEarthNet
*class names*; this one retrieves real BigEarthNet.txt *question/answer
records*. Both are prompt-time context, neither changes model weights.

DATASET SOURCE: BIFOLD-BigEarthNetv2-0/BigEarthNet.txt on Hugging Face
(https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt) -
464,044 co-registered Sentinel-1/Sentinel-2 pairs, ~9.6M textual
annotations (captions, VQA pairs, referring-expression instructions),
columns include id, input, output, type, category, split, country, season,
climate_zone. This module never downloads the corresponding satellite
image corpus - only a small local subset of the *text* columns.

WHY A LOCAL CACHE, NOT A LIVE DOWNLOAD AT IMPORT TIME: this module does
not reach out to the network itself. It reads a small, pre-prepared local
subset (a CSV/TSV/TXT file with at least `input`, `output`, `split`
columns) from `data/cache/`. Preparing that subset - fetching it from
Hugging Face once, filtering to a few hundred to ~2000 real `train`-split
rows, and saving them locally - is a one-time, out-of-band step (see
`docs/rs_adaptation.md` for exact instructions), not something `run()`
does per query. If no cache file is present, `load_dataset_subset()`
honestly returns `None` and every caller downstream degrades gracefully -
this module NEVER fabricates example records to fill the gap.

TRAIN-SPLIT ONLY: to keep this a source of domain *context* rather than
any risk of evaluation leakage, `load_dataset_subset()` keeps only rows
whose `split` column equals `"train"` (case-insensitive) and discards
everything else (`test`, `validation`, `bench`, or anything unrecognized).

RETRIEVAL METHOD: TF-IDF (scikit-learn's `TfidfVectorizer`, already a
project dependency) + cosine similarity - no embedding API, no external
LLM, no vector database. The index is built once per process (module-level
cache, mirroring the pattern `specialists/vqa_smolvlm.py:_load()` uses for
the model itself) and reused across queries, so per-query retrieval is a
single small matrix-vector product - well under the 1-second budget for a
corpus of a few thousand rows.
"""
from __future__ import annotations

import csv
import dataclasses
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- locating the local cache -------------------------------------------

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]  # src/specialists/this_file.py -> repo root

# Accepted, in priority order. A .parquet cache is honored only if pandas
# *and* a parquet engine happen to be installed (soft-optional - this
# project does not add pandas/pyarrow as hard dependencies for this one
# feature). CSV/TSV/TXT need only the Python standard library.
DEFAULT_CACHE_CANDIDATES: Tuple[str, ...] = (
    "data/cache/bigearthnet_txt_subset.csv",
    "data/cache/bigearthnet_txt_subset.tsv",
    "data/cache/bigearthnet_txt_subset.txt",
    "data/cache/bigearthnet_txt_subset.parquet",
)

ENV_OVERRIDE = "SATQUERY_BIGEARTHNET_TXT_CACHE"

REQUIRED_COLUMNS = {"input", "output", "split"}

DATASET_SOURCE = "BigEarthNet.txt"
DATASET_SOURCE_FULL = (
    "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt "
    "(huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt)"
)


@dataclasses.dataclass(frozen=True)
class RSExample:
    """One real BigEarthNet.txt text record kept for retrieval."""

    record_id: str
    input: str
    output: str
    type: str = ""
    category: str = ""
    split: str = "train"


def _candidate_paths() -> List[Path]:
    override = os.environ.get(ENV_OVERRIDE)
    paths = [Path(override)] if override else []
    paths += [_REPO_ROOT / p for p in DEFAULT_CACHE_CANDIDATES]
    return paths


def _rows_from_delimited(path: Path) -> List[Dict[str, str]]:
    # Sniff the delimiter rather than assuming comma - the upstream file is
    # literally named "BigEarthNet.txt", so a tab- or comma-separated
    # export both need to work without the caller renaming anything.
    with open(path, "r", encoding="utf-8", newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(fh, dialect=dialect)
        return [dict(row) for row in reader]


def _rows_from_parquet(path: Path) -> Optional[List[Dict[str, str]]]:
    try:
        import pandas as pd  # optional; only used if already installed
    except ImportError:
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:  # noqa: BLE001 - any parquet-engine failure -> treat as unavailable, never crash the caller
        return None
    return df.astype(str).to_dict("records")


def load_dataset_subset(path: Optional[str] = None) -> Optional[List[RSExample]]:
    """Loads the local BigEarthNet.txt subset, `train`-split rows only.

    Returns None (never an empty-but-fake list, never raises to the
    caller) if no cache file is found, the file has none of the required
    columns, or every row is filtered out. This is the single honesty
    boundary for the whole module: everything downstream treats `None` as
    "dataset-backed adaptation unavailable" and degrades accordingly.
    """
    candidates = [Path(path)] if path else _candidate_paths()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            if candidate.suffix.lower() == ".parquet":
                rows = _rows_from_parquet(candidate)
                if rows is None:
                    continue
            else:
                rows = _rows_from_delimited(candidate)
        except (OSError, UnicodeDecodeError, csv.Error):
            continue

        if not rows:
            continue
        header = {k.strip().lower() for k in rows[0].keys()}
        if not REQUIRED_COLUMNS.issubset(header):
            continue

        examples: List[RSExample] = []
        for i, row in enumerate(rows):
            norm = {k.strip().lower(): (v or "").strip() for k, v in row.items()}
            if norm.get("split", "").strip().lower() != "train":
                continue
            if not norm.get("input") or not norm.get("output"):
                continue
            examples.append(RSExample(
                record_id=norm.get("id") or norm.get("record_id") or str(i),
                input=norm["input"],
                output=norm["output"],
                type=norm.get("type", ""),
                category=norm.get("category", ""),
                split="train",
            ))
        if examples:
            return examples
    return None


# --- retrieval index (built once, cached at module level) ---------------

class RSExampleIndex:
    """Deterministic TF-IDF + cosine-similarity retrieval over a fixed set
    of real BigEarthNet.txt `train`-split examples. No embeddings, no
    external calls - `sklearn.feature_extraction.text.TfidfVectorizer` is
    already a project dependency (used nowhere near a network at fit or
    query time)."""

    def __init__(self, examples: List[RSExample]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.examples = examples
        # Indexed on input+output together (per spec: "Index the real
        # `input` and `output` text fields") so a query can match either
        # the phrasing of similar questions or the vocabulary of their
        # answers.
        corpus = [f"{ex.input} {ex.output}" for ex in examples]
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        self._matrix = self._vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 3) -> List[RSExample]:
        """Returns up to `top_k` examples most similar to `query`, most
        similar first. Fully deterministic (TF-IDF + cosine similarity has
        no randomness). Returns an empty list - never a fabricated one -
        if the query shares no vocabulary with the indexed corpus."""
        from sklearn.metrics.pairwise import cosine_similarity

        query = (query or "").strip()
        if not query or not self.examples:
            return []
        top_k = max(0, min(top_k, 3))
        if top_k == 0:
            return []
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        ranked = sorted(range(len(sims)), key=lambda i: (-sims[i], i))
        return [self.examples[i] for i in ranked[:top_k] if sims[i] > 0]


_INDEX_CACHE: Dict[str, Optional[RSExampleIndex]] = {}


def get_index(cache_key: str = "default") -> Optional[RSExampleIndex]:
    """Builds (once per process) and returns the retrieval index, or None
    if no usable local dataset subset is present. Cached so the TF-IDF fit
    - the only non-trivial cost here - never re-runs per query."""
    if cache_key not in _INDEX_CACHE:
        examples = load_dataset_subset()
        _INDEX_CACHE[cache_key] = RSExampleIndex(examples) if examples else None
    return _INDEX_CACHE[cache_key]


def dataset_records_available(cache_key: str = "default") -> int:
    index = get_index(cache_key)
    return len(index.examples) if index else 0


def build_domain_context_block(examples: List[RSExample]) -> str:
    """Formats retrieved examples as a compact, clearly-labeled context
    block. Explicitly instructs the model to use the image, not copy the
    reference answers - these are real BigEarthNet.txt records, but they
    describe *different* images than the one the user uploaded."""
    if not examples:
        return ""
    lines = [
        "REMOTE-SENSING DOMAIN CONTEXT (reference examples from other "
        "BigEarthNet.txt images - NOT the uploaded image; do not copy "
        "these answers, use them only to calibrate the kind of answer "
        "expected; answer strictly from the image actually provided):",
    ]
    for i, ex in enumerate(examples, start=1):
        lines.append(f"Example {i}:")
        lines.append(f"  Question/Instruction: {ex.input}")
        lines.append(f"  Reference answer: {ex.output}")
    return "\n".join(lines)


def retrieve_examples(query: str, top_k: int = 3, cache_key: str = "default") -> List[RSExample]:
    """Convenience entry point used by `specialists/vqa_smolvlm.py`.
    Returns [] whenever the dataset subset isn't locally available -
    never raises, never fabricates."""
    index = get_index(cache_key)
    if index is None:
        return []
    return index.retrieve(query, top_k=top_k)
