# AERIS AI

**Agentic Earth Observation Reasoning & Intelligence System**

*Submission Details:*
- **Problem Statement**: SIH26209
- **Theme**: Space Technology
- **Category**: Software
- **Team**: Mind Map Cache
- **Team ID**: 161454

*Repository*: https://github.com/CodyRohith7/aeris-ai-sih26209

## Demo

AERIS AI Demo:
https://youtu.be/Xl34KnMzDeA

## Overview

AERIS AI is an agentic Earth observation analysis system that converts natural-language queries and satellite imagery into routed, evidence-backed analysis across VQA, grounding, bi-temporal change detection, and optical + SAR workflows.

Given one or two satellite/aerial images and a natural-language question, it validates the input, routes the query to the appropriate analysis capability, executes it, generates visual evidence, computes an honest confidence score, and produces a full execution trace plus an exportable report (JSON and PDF) — through the Streamlit "AERIS AI — Mission Console" UI or a CLI. Two real, additive prompt-time adaptation layers inject genuine BigEarthNet-derived remote-sensing context into the VQA prompt (see [Remote-Sensing Adaptation](#remote-sensing-adaptation)).

The system's guiding principle is **honesty about what actually ran**:
- **VQA** uses real `HuggingFaceTB/SmolVLM-256M-Instruct` vision-language model inference on CPU.
- **Grounding, change detection, and optical + SAR analysis** are classical specialist baselines (computer vision algorithms), not trained foundation models.
- Every confidence score measures computed signal or generation certainty rather than calibrated factual-correctness probability.

## Problem

Earth-observation analysis is fragmented across specialized, task-specific tools — a user who wants to ask a question about imagery, locate a feature, compare two dates, or combine optical and SAR signal today needs to understand which tool applies, how to prepare its inputs, and how to interpret its output, with no single interface, no shared evidence standard, and no visibility into which analysis actually ran or how much to trust it.

Problem Statement context: **SIH26209** (Student Innovation / Space Technology / Software), calling for an AI assistant capable of interpreting remote-sensing imagery (single-image question answering, object/feature grounding, bi-temporal change detection, and optical+SAR fusion) with a generic, non-remote-sensing-adapted vision-language model explicitly called out as insufficient for the domain.

## Solution / Innovation

AERIS AI implements one natural-language interface over four real Earth-observation analysis capabilities — ingestion, validation, intent routing, specialist execution, evidence generation, confidence scoring, execution tracing, and export — using a combination of a real pretrained vision-language model (for single-image VQA) and classical computer-vision baselines (for grounding, change detection, and fusion), with every component's real nature disclosed rather than implied, and every step of the routing/execution decision surfaced as an observable "AERIS Orchestration" trace rather than a black box. Weight-level remote-sensing-domain adaptation (the work that would close the gap between "a generic pretrained VLM" and a genuinely RS-adapted one) was designed, evaluated, and documented in detail (see `docs/rs_adaptation.md`), but was not completed — larger candidate models were found impractical on the available CPU-only hardware. Two lightweight, prompt-time-only adaptation layers ARE implemented and real (see [Remote-Sensing Adaptation](#remote-sensing-adaptation) below) — neither trains or changes any model weight.

**Use cases**: agriculture (land-cover and vegetation analysis), urban monitoring (built-up and infrastructure analysis), water (water-region identification and monitoring), change analysis (bi-temporal investigation), infrastructure and environmental observation — the Mission Console's mission-preset buttons map to these areas as UX shortcuts onto the same four real specialists, not domain-specific trained models.

## Technical Honesty

- **VQA** = real pretrained model (`HuggingFaceTB/SmolVLM-256M-Instruct`), run on CPU.
- **Grounding / change / fusion** = current classical computer-vision baselines, not trained deep-learning models.
- **Dataset-backed adaptation** = the retrieval architecture is implemented and tested end-to-end (`src/specialists/rs_example_adapter.py`); live BigEarthNet.txt records are currently unavailable in every reachable environment (`huggingface.co` blocked by organization egress policy — see `docs/rs_adaptation.md` §8), so it honestly reports `applied: false` until a local subset is supplied.
- **Benchmark scores** = not claimed. `src/evaluation/benchmarks.py` marks every benchmark `NOT_YET_EVALUATED`.

## Key Capabilities

1. **Single-image VQA** — Answers a natural-language question about one image. Real model: `HuggingFaceTB/SmolVLM-256M-Instruct` (`tool_single_image_vqa_smolvlm_v1`), a genuine, general-purpose vision-language model — **not** remote-sensing-domain-adapted — run on CPU. Falls back to a classical color/texture clustering baseline (`tool_single_image_vqa_v0`) if the model is unavailable or fails, with the fallback fully disclosed in the output. Two prompt-time-only context layers are applied before generation — see [Remote-Sensing Adaptation](#remote-sensing-adaptation) — neither changes the model's weights. **Limitation**: a small (256M-parameter), general-purpose VLM trades some accuracy for CPU-feasibility, and it has no remote-sensing-specific *training*.
2. **Grounding** — Locates a described feature in an image and draws a bounding box. Method: classical computer vision (`tool_grounding_v0` — HSV color thresholding + contour extraction), not a deep-learning or foundation-model detector. **Limitation**: two real foundation-model candidates (Florence-2-base, `grounding-dino-tiny`) were evaluated on real hardware and both collapsed to near-full-image boxes on this kind of imagery — a documented model/domain limitation, not a bug — so the classical baseline remains the shipped capability (see `docs/rs_adaptation.md`).
3. **Bi-temporal change detection** — Compares two images of the same scene at different dates and highlights what changed. Method: classical CV (`tool_change_v0` — grayscale differencing + Otsu thresholding + morphological cleanup), not a trained change-detection model. **Limitation**: sensitive to illumination/registration differences between the two images that aren't real scene change.
4. **Optical+SAR fusion** — Jointly analyzes an optical and a SAR image of the same scene. Method: classical CV (`tool_fusion_v0` — k-means clustering over a stacked `[R, G, B, SAR]` feature vector), not a trained multi-modal fusion model. **Limitation**: no learned cross-modal representation; a hand-built joint feature space only.
5. **Intelligent routing** — A rule-based router combined with a trained intent classifier picks the right capability from the query text and the declared inputs (single image vs. pair, modality, dates), and explicitly refuses to guess (`needs_clarification`) when the query is genuinely ambiguous. **Limitation**: rule-based routing can misroute unusual phrasing; it is not a learned end-to-end agent.
6. **Evidence generation** — Every result includes a real, computed visual evidence artifact (bounding box overlay, change-map composite, fused cluster map, or the source image for VQA) — never a placeholder image. **Limitation**: evidence quality is bounded by the same classical-CV or small-VLM limitations described above.
7. **Confidence scoring** — Every result carries a confidence score derived from a real, specialist-specific computed signal (documented formula in `docs/confidence.md`), explicitly labeled by `method_version` (`v0_classical` heuristic signal vs. `v1_vlm_mean_token_probability` for the real model) so it is never mistaken for a calibrated correctness probability. **Limitation**: neither method version is a calibrated "this answer is factually correct" probability — that would require ground-truth-labeled evaluation data this project does not have.
8. **Execution trace** — Every run produces a complete, inspectable `ExecutionTrace` (query → validation → routing decision → specialist output → confidence → evidence), shown as a human-readable timeline and as raw JSON, both in the UI and in exports. **Limitation**: the trace reflects one pipeline run; it is not a persisted multi-session audit log.
9. **Contextual follow-up** — A follow-up question re-runs the same real pipeline against the same cached input image(s) without re-uploading, and keeps a real conversation history of prior (question, answer) pairs. **Limitation**: this is independent re-querying, not a memory-augmented conversational model — the VLM itself has no persistent context across turns.
10. **PDF + JSON reporting** — Exports a ten-section PDF report (Analysis Summary, Input Data, Analysis Result, Visual Evidence, Active Specialist, Confidence, Agent Execution Timeline, Technical Trace, Limitations & Warnings, Export Metadata) and the raw JSON trace, both generated directly from the same `ExecutionTrace` object rendered on screen. **Limitation**: PDF generation requires `reportlab`; without it, only JSON export is available.

## Architecture

```
Input (image(s) + query)
   │
   ▼
Validation      — src/ingestion/  (raster load, metadata, bi-temporal/cross-modal checks)
   │
   ▼
Router          — src/routing/router.py + intent_classifier.py
   │
   ▼
Specialist      — src/specialists/, src/grounding/, src/change/, src/fusion/
   │
   ▼
Evidence        — src/evidence/composer.py
   │
   ▼
Confidence      — src/confidence/engine.py
   │
   ▼
Execution Trace — src/routing/schemas.py (ExecutionTrace)
   │
   ▼
Report          — src/export/report.py (JSON + PDF)
```

`app/pipeline.py:run_query()` is the single orchestration path both `app/cli.py` and `app/streamlit_app.py` call — there is no separate UI-specific reimplementation of routing/validation/dispatch logic.

## Project Structure

```
aeris-ai/
├── app/                  # CLI (cli.py) and Streamlit UI (streamlit_app.py) + shared pipeline.py
├── src/
│   ├── ingestion/        # raster I/O, metadata inspection, input validation
│   ├── routing/          # router, intent classifier, schemas, timeline, failure classification
│   ├── specialists/      # VQA (classical + SmolVLM dispatch), RS context & example adapters
│   ├── grounding/        # classical grounding baseline
│   ├── change/           # classical bi-temporal change baseline
│   ├── fusion/           # classical optical+SAR fusion baseline
│   ├── evidence/         # evidence-image composition
│   ├── confidence/       # confidence engine
│   ├── models/           # model/tool availability registry, Florence-2 compat shim
│   ├── export/           # JSON + PDF report generation
│   └── evaluation/       # benchmark interfaces (VRSBench/RSVQA/CDVQA/ISRO-SAC — NOT_YET_EVALUATED)
├── demo_assets/          # real Sentinel-1/Sentinel-2 demo assets and provenance records
├── scripts/              # one-off verification/research scripts (see docs/rs_adaptation.md)
├── configs/              # rs_adaptation.yaml (not-yet-executed LoRA adaptation config)
├── data/
│   ├── fixtures/         # synthetic, seeded demo images (see data/fixtures/README.md)
│   ├── manifests/        # real-dataset manifest template (none downloaded yet)
│   ├── processed/, raw/, samples/   # local storage for imagery
├── docs/                 # architecture/design/research documentation, evidence/
├── tests/                # unit/ and integration/ test suites
├── requirements.txt              # core runtime dependencies
├── requirements-optional.txt     # rasterio + research-script-only dependencies
└── .gitignore
```

## Installation

Tested on **Windows with Miniforge/Conda, Python 3.10.x**:

```powershell
conda create -n aeris python=3.10 -y
conda activate aeris
cd C:\path\to\aeris-ai
pip install -r requirements.txt
```

Install `requirements-optional.txt` only if you need real GeoTIFF I/O (`rasterio`) or want to re-run the research scripts in `scripts/`:

```powershell
pip install -r requirements-optional.txt
```

## Run

```powershell
conda activate aeris
streamlit run app\streamlit_app.py
```

Opens the **AERIS AI — Mission Console** at `http://localhost:8501`.

The Mission Console provides five tabs covering unified and specialist workflows:

1. **🛰️ · ASK AERIS (Unified Mission Console)**: The primary entry point. Upload 1 image (for single-image VQA or grounding) or 2 images (for bi-temporal change or optical+SAR fusion), or select from bundled synthetic fixtures or real Sentinel-1/Sentinel-2 demo presets. Enter any natural-language query; AERIS validates the sensor inputs, infers intent, automatically routes to the appropriate specialist, presents a real-time execution plan before the result, displays visual evidence, computes signal-based confidence, and shows an agent process timeline with "Why this result?" rationale.
2. **01 · SINGLE IMAGE**: Dedicated specialist tab for single-image VQA using `HuggingFaceTB/SmolVLM-256M-Instruct` with prompt-time domain context adaptation (falls back to classical color/texture heuristics if unavailable).
3. **02 · GROUNDING**: Dedicated specialist tab for spatial feature grounding using classical computer-vision contour extraction and HSV thresholding to draw bounding boxes and calculate area coverage.
4. **03 · BI-TEMPORAL CHANGE**: Dedicated specialist tab for comparing before/after image pairs to detect and highlight land-cover changes via grayscale differencing, Otsu thresholding, and morphological filtering.
5. **04 · OPTICAL + SAR**: Dedicated specialist tab for cross-modal analysis fusing optical RGB and synthetic aperture radar (SAR) amplitude data via k-means feature clustering to delineate water bodies and urban built-up areas.

A CLI is also available:

```powershell
python app\cli.py --image1 "C:\path\to\your_image.jpg" --query "What is the dominant land cover or feature visible in this image?"
```

## Example Workflow

1. In the **🛰️ · ASK AERIS** tab, select a sample preset (e.g. *AERIS Demo: VQA Mission (real Sentinel-2 optical)* or bundled demo fixture) or upload an image.
2. Enter your query: *"What is the dominant land cover or feature visible in this image?"*
3. Click **Analyze**. The Mission Console renders the real-time **Plan** breadcrumb (`QUERY → INPUT CHECK → INTENT DETECTED → SPECIALIST SELECTED → ANALYSIS → EVIDENCE → RESULT`), executes the pipeline, and displays the answer alongside the source visual evidence.
4. Inspect the **Reliability** and **Sensor Intelligence** cards, the **Agent process timeline**, and the **Why this result?** explanation.
5. Ask a follow-up question (e.g. *"Is there visible water in this scene?"*) without re-uploading — the conversation history expands to track the exchange.
6. Export the result as an official **PDF report** or structured **JSON** execution trace.

## Exports

- **JSON** — the complete raw `ExecutionTrace` (query, validation, routing decision, specialist output, confidence, evidence references).
- **PDF report** — a ten-section formatted report (Analysis Summary, Input Data, Analysis Result, Visual Evidence, Active Specialist, Confidence, Agent Execution Timeline, Technical Trace, Limitations & Warnings, Export Metadata) with page numbers and a running header/footer.

## Testing

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Verified in the local sandbox mirror (classical/CPU-fallback paths only, no network egress for model downloads or dataset access in that environment):
**136 tests collected, 122 executed and passed, 14 skipped, 0 failed.**
The 14 skips are tests gated on `torch`/`transformers`/`streamlit` being importable (real-model and real-Streamlit `AppTest` smoke tests), plus one gated on a local BigEarthNet.txt cache being present (see [Remote-Sensing Adaptation](#remote-sensing-adaptation)) — on a machine with the full `requirements.txt` installed and that cache prepared (e.g. the target Windows/conda environment), those run for real instead of skipping.

## Performance

Measured on real CPU hardware (target Windows machine, no GPU):

- **VQA (SmolVLM-256M-Instruct) model load**: ~47 seconds.
- **VQA inference**: typically **~35–60 seconds** per query on CPU; one measured run took **128.47 seconds**, attributed to CPU contention on that run rather than a regression (see the project's engineering history for the investigation). This is **not real-time inference** — plan for tens of seconds of latency per VQA query.
- **Grounding/change/fusion (classical CV)**: sub-second to a few seconds, since these do not load a neural network.

## Remote-Sensing Adaptation

AERIS AI distinguishes four clearly separate things, and never blurs one into another:

**A. Current VQA model** — `HuggingFaceTB/SmolVLM-256M-Instruct` (`src/specialists/vqa_smolvlm.py`), the stock pretrained checkpoint, run as-is on CPU. Its weights are never modified by anything below.

**B. Dataset-grounded adaptation (implemented)** — two independent, additive, prompt-time-only context layers, both applied before generation and neither touching a single model weight:

  1. *Taxonomy context* (`src/specialists/rs_context_adapter.py`) — the public BigEarthNet-19 land-cover class-name list (19 fixed strings, cited, not downloaded), keyword-matched into the prompt when relevant.
  2. *Retrieval-augmented context* (`src/specialists/rs_example_adapter.py`) — real BigEarthNet.txt text records (question/instruction + reference-answer pairs, from [BIFOLD-BigEarthNetv2-0/BigEarthNet.txt](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt)) retrieved by TF-IDF + cosine similarity from a small local cache (target: a few hundred to ~2000 `train`-split rows only — never the 464k-pair satellite image corpus). The 2-3 most similar real records are appended to the prompt as explicitly-labeled reference examples, with the model told not to copy their answers and to answer only from the actual uploaded image.

  **Preparing the local cache**: this repository does not ship a BigEarthNet.txt subset (the dataset is ~9.6M rows; only a small filtered slice is needed). Download the dataset's metadata once from a machine with real internet access, filter to `split == "train"`, keep the `id`/`input`/`output`/`type`/`category`/`split` columns, sample roughly 500–2000 rows, and save as `data/cache/bigearthnet_txt_subset.csv` (or point `AERIS_BIGEARTHNET_TXT_CACHE` at another path). Without that file present, this layer honestly reports itself as unavailable — see "Graceful fallback" below — rather than fabricating examples.

  **Trace metadata** — every real-model VQA result's `raw` dict carries both layers' status separately: `rs_context_adaptation` (taxonomy) and `rs_example_adaptation` (retrieval — `applied`, `method`, `source`, `dataset_records_available`, `examples_retrieved`, `fine_tuned` (always `False`), `retrieved_ids`). The Streamlit UI shows a matching "REMOTE-SENSING ADAPTATION" indicator next to the VQA result whenever either layer actually fired.

  **Graceful fallback** — if no local cache is present, or loading/parsing it fails for any reason, `rs_example_adaptation.applied` is `False`, `dataset_records_available` is `0`, VQA proceeds completely normally (the taxonomy layer and the base query are unaffected), and the UI shows "Dataset-backed adaptation unavailable" rather than staying silent about it.

**C. NOT implemented** — parameter fine-tuning, LoRA, or any other form of weight adaptation. Nothing in this repository trains SmolVLM on BigEarthNet or any other remote-sensing dataset. The earlier, more ambitious LoRA-adaptation plan (targeting a larger RS-oriented VLM) was designed and documented but never executed — see [Research / Model Notes](#research--model-notes) and `docs/rs_adaptation.md`.

**D. Future work** — actual parameter-level fine-tuning of a remote-sensing-adapted VLM, and benchmark evaluation against VRSBench / RSVQA / CDVQA once real, held-out ground truth is available (see `src/evaluation/benchmarks.py`'s `NOT_YET_EVALUATED` status).

## Limitations

- **Grounding, change detection, and fusion are classical computer-vision baselines**, not deep-learning or foundation-model-based — two real foundation-model grounding candidates were tried on real hardware and both failed to produce spatially meaningful boxes on this kind of imagery (a documented model/domain limitation).
- **The VQA model's weights are not remote-sensing-domain-adapted** — it is the stock pretrained SmolVLM-256M-Instruct checkpoint. Two prompt-time context layers (a fixed taxonomy and BigEarthNet.txt-record retrieval, see [Remote-Sensing Adaptation](#remote-sensing-adaptation)) are real but do not change this; a LoRA adaptation plan targeting a larger remote-sensing-oriented VLM was designed and documented but not executed (see `docs/rs_adaptation.md`).
- **CPU-only inference is slow** (tens of seconds per VQA query, ~47s model load) — there is no GPU acceleration path in the current setup.
- **Small VLM size (256M parameters)** trades some answer accuracy and robustness for CPU feasibility.
- **No calibrated correctness probability** — confidence scores measure signal quality (classical) or generation certainty (VLM), never factual-correctness probability, since no ground-truth-labeled evaluation data exists for this project yet (`NOT_YET_EVALUATED` in `src/evaluation/benchmarks.py`).
- **No real satellite imagery is bundled** — the fixtures in `data/fixtures/` are synthetic, seeded demo images, clearly labeled as such (`data/fixtures/README.md`); results on them describe the synthetic pattern, not a real place.
- **Optional `rasterio` dependency** — without it, raster loading falls back to Pillow (PNG/JPEG only, no CRS/band metadata); real GeoTIFF ingestion with full geospatial metadata requires installing `requirements-optional.txt`.

## Research / Model Notes

Larger remote-sensing-oriented VLM candidates — notably **PaliGemma 2** (`google/paligemma2-3b-pt-448`, the original LoRA-adaptation target in `configs/rs_adaptation.yaml`) — were evaluated and ultimately not pursued as impractical for the available CPU-only hardware (no GPU, limited RAM). Two general-purpose grounding foundation models (Florence-2-base, `grounding-dino-tiny`) were also evaluated as grounding-capability upgrades and rejected after real-hardware testing showed both collapsing to near-full-image boxes on this kind of imagery. The full evaluation history, real measured numbers, and the reasoning behind each decision are in `docs/rs_adaptation.md`; the exact commands used to reproduce every step on real hardware are in `docs/RUN_ON_WINDOWS.md`.

## License

**No license currently exists for this repository.** No `LICENSE` file is present. Until one is added, this code is not licensed for reuse, modification, or redistribution beyond what applies by default under copyright law — please add a license before treating this repository as open source.
