#!/usr/bin/env python3
"""AERIS AI - Agentic Earth Observation Reasoning & Intelligence System.
Mission Console UI (Streamlit).

Product identity: this application was developed as "SatQuery AI" during
SIH26167-scoped development (that history is preserved in code comments,
docs/, and changelog references below) and is now submitted front-facing
as AERIS AI under Student Innovation / Space Technology / Software
(SIH26209). This is a rebrand and UX pass only - see the "PRODUCT PIVOT"
note below for exactly what did and did not change.

PRODUCT PIVOT (kept brief; the substance is unchanged): every rendering
function below still calls the exact same `pipeline.run_query()` / renders
the exact same `ExecutionTrace` objects it always did (see the
architectural rule two paragraphs down, still true). The pivot adds: a
"Mission Console" framing for the unified analysis tab, mission-preset
buttons that populate a suggested query (same mechanism as the pre-existing
example-query chips), a compact "Mission Brief" summary card built from
real trace/output fields, a prominent "AERIS Orchestration" relabeling of
the existing, real `routing/timeline.py` step checklist, per-specialist
"Analysis cards" that surface the same REAL MODEL / CLASSICAL / UNAVAILABLE
status the sidebar already computes, and demo-preset sample-data entries
that point at the real local Sentinel-1/Sentinel-2 derived PNGs in
`demo_assets/REAL_OPTICAL_SAR/` alongside the pre-existing synthetic
fixtures. No routing, validation, specialist, confidence, or evidence logic
changed.

Design plan and rationale: docs/ui_design.md - see "Revision 4" for the
FINAL UI REFINEMENT (visual-only) pass this file implements on top of
"Revision 3"'s features (agent process timeline, analysis registry,
conversational follow-up, numeric error codes, richer input guidance).
Revision 4 changes composition, color roles, and information architecture
(header status, vertical timeline, transcript-style conversation, a
functional "Return to analysis" recovery action) - it does NOT change
routing, validation, specialist dispatch, confidence, or evidence
computation. The two small, additive backend helpers from Revision 3
(`routing/timeline.py`, `routing/failure_classification.py`) are pure,
read-only summarizations of fields an `ExecutionTrace` already carries;
see their docstrings.

Architectural rule this file MUST NOT break: it calls the exact same
orchestration function the CLI calls - `pipeline.run_query()` - and renders
the exact same `ExecutionTrace` / `SpecialistOutput` / `ConfidenceResult`
objects the CLI already prints. There is no UI-specific reimplementation of
routing, validation, dispatch, confidence, or evidence logic anywhere below;
this module is presentation only. Every number shown on screen is read
directly off those real objects - short display labels (e.g. "Generation
certainty" for method_version "v1_vlm_mean_token_probability") are a fixed,
documented lookup over the small set of real values the system can ever
produce (see docs/confidence.md) - the full, verbatim real text is always
one click away in a "Details" expander, never deleted. The "Agent process
timeline" checklist and the numeric error-code badges are likewise derived
strictly from real trace fields (see routing/timeline.py and
routing/failure_classification.py) - nothing here is a fake progress
animation or an invented status.

Run with:  streamlit run app/streamlit_app.py

Not runnable in the cloud development sandbox this project was largely
built in (no network egress to install streamlit there - see
docs/network_constraints.md); it must be run and verified on real hardware.
See docs/RUN_ON_WINDOWS.md.
"""
from __future__ import annotations

import datetime
import html
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st  # noqa: E402

import pipeline  # noqa: E402 - shared orchestration, same module app/cli.py uses
from export import report  # noqa: E402
from evidence import composer  # noqa: E402
from ingestion import raster_io  # noqa: E402
from ingestion import metadata as metadata_mod  # noqa: E402
from models import registry  # noqa: E402
from routing import failure_classification as failure_mod  # noqa: E402
from routing import timeline as timeline_mod  # noqa: E402
from routing.schemas import ExecutionTrace, SpecialistOutput  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
FIXTURES_DIR = os.path.join(REPO_ROOT, "data", "fixtures")
RUNS_DIR = os.path.join(REPO_ROOT, "runs", "streamlit")
SMOLVLM_TOOL_NAME = "tool_single_image_vqa_smolvlm_v1"
CLASSICAL_VQA_TOOL_NAME = "tool_single_image_vqa_v0"
BUILD_LABEL = "v0 (MVP baseline)"

# Plain-English labels for the real task_type strings - copy, not data.
TASK_TYPE_LABELS = {
    "single_image_vqa": "Single-image VQA",
    "grounding": "Feature grounding",
    "bitemporal_change": "Bi-temporal change",
    "optical_sar_fusion": "Optical + SAR fusion",
}

# Static per-tab navigation copy (number, title, one-line descriptor) - not
# derived from anything, just the fixed labels for the four real
# capabilities this system has. Streamlit's native tab labels are a single
# line with no subtext, so the descriptor is rendered as a subtitle under
# each tab's own "Input" section header instead of inside the tab strip
# itself (see docs/ui_design.md Revision 4).
TAB_INFO = {
    "vqa": ("01", "Single image", "Ask & interpret"),
    "gr": ("02", "Grounding", "Locate features"),
    "ch": ("03", "Bi-temporal change", "Compare acquisitions"),
    "fu": ("04", "Optical + SAR", "Fuse modalities"),
}

# Fixed, exhaustive short labels for the two method_version values the
# confidence engine can ever produce (see docs/confidence.md) - not a
# per-result invention, just a compact name for a documented category. The
# full basis_description is always shown verbatim in the Details expander.
CONFIDENCE_SHORT_LABEL = {
    "v0_classical": "Signal quality heuristic",
    "v1_vlm_mean_token_probability": "Generation certainty",
}
CONFIDENCE_CAVEAT = {
    "v0_classical": "Reflects classical-algorithm signal quality, not a calibrated model probability.",
    "v1_vlm_mean_token_probability": "Reflects how confidently the model generated its wording, not factual correctness.",
}

# Compact, real (non-marketing) per-mode guidance - what to provide, why the
# analysis needs it, accepted formats, and what happens next. Keeps the
# input area to one always-visible line plus an opt-in expander instead of a
# wall of text (see docs/ui_design.md Revision 3).
MODE_HELP: Dict[str, Dict[str, str]] = {
    "vqa": {
        "provide": "One image and a written question about it.",
        "why": "The vision-language model (or classical fallback) answers directly from pixel content - there is no separate metadata lookup.",
        "format": "PNG, JPEG, or GeoTIFF/TIFF.",
        "happens": "You get a plain-text answer, a confidence score, and the source image shown as evidence.",
    },
    "gr": {
        "provide": "One image and a query naming a target (e.g. water, vegetation, urban).",
        "why": "The classical grounding baseline segments and localizes the named target type - it does not recognize arbitrary open-vocabulary objects.",
        "format": "PNG, JPEG, or GeoTIFF/TIFF.",
        "happens": "Detected regions are boxed on the image and reported with a confidence score.",
    },
    "ch": {
        "provide": "Two images of the same location at different times, ideally with acquisition dates.",
        "why": "Change detection compares pixel intensity between the two dates - it needs the pair to differ in time, not just in content.",
        "format": "PNG, JPEG, or GeoTIFF/TIFF, same or similar extent.",
        "happens": "You get a before/after/change-highlighted view and the percentage of the scene that changed.",
    },
    "fu": {
        "provide": "One optical image and one SAR image of the same location.",
        "why": "Fusion combines complementary optical and radar signal to separate built-up, water, and vegetated areas more robustly than either alone.",
        "format": "PNG, JPEG, or GeoTIFF/TIFF.",
        "happens": "You get a fused cluster map alongside the two raw inputs shown for reference.",
    },
}

# Example queries per mode - clickable starter prompts that only fill the
# query field (via a callback that sets session_state before the text_input
# widget reads it); they never auto-run analysis themselves.
EXAMPLE_QUERIES: Dict[str, List[str]] = {
    "vqa": [
        "Describe the major land cover types visible in this image.",
        "Is there any visible water in this image?",
        "What is the dominant land use in this scene?",
    ],
    "gr": [
        "Locate the water body in this image and highlight it.",
        "Find the vegetated areas in this image.",
        "Highlight the urban / built-up region.",
    ],
    "ch": [
        "What changed between these two dates?",
        "Did the built-up area expand?",
        "Compare vegetation cover before and after.",
    ],
    "fu": [
        "Use both images to identify built-up and water-covered regions.",
        "Where does the SAR signal disagree with the optical image?",
    ],
    "ask": [
        "Describe the major land cover types visible in this image.",
        "Locate the water body in this image and highlight it.",
        "What changed between these two dates?",
        "Use both images to identify built-up and water-covered regions.",
        "Is there any visible water in this image?",
    ],
}

_STATUS_CLASS = {
    timeline_mod.STATUS_DONE: "sq-tl-done",
    timeline_mod.STATUS_FAILED: "sq-tl-failed",
    timeline_mod.STATUS_NOT_REACHED: "sq-tl-pending",
}
# A textual fallback for the two non-"done" states, so status is never
# color-only (accessibility) - "done" needs no extra word since a step's
# own detail (latency, item count, etc.) already confirms it ran.
_STATUS_FALLBACK_TEXT = {
    timeline_mod.STATUS_FAILED: "Failed",
    timeline_mod.STATUS_NOT_REACHED: "Not reached",
}
_BACKEND_STATE_COLOR = {
    "REAL MODEL": "var(--accent2)",
    "CLASSICAL": "var(--text-dim)",
    "UNAVAILABLE": "var(--text-faint)",
}

st.set_page_config(
    page_title="AERIS AI - Mission Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design tokens / CSS - see docs/ui_design.md "Revision 3". Targets
# Streamlit's data-testid attributes, which are reasonably stable across
# recent versions but can shift between major releases; if a widget renders
# unstyled on your installed Streamlit version, the fix is to adjust the
# selector below, not to add a competing style system. No external font
# dependency - the app must look correct with no internet access.
#
# Two real, user-reported visual bugs from the previous pass are addressed
# here on a best-effort basis (this cannot be run in a real browser in this
# sandbox - see docs/RUN_ON_WINDOWS.md, please re-screenshot to confirm):
#   1. Sidebar "Technical system status" text wrapping character-by-character
#      -> registry rows are now stacked (label above, value below, full
#         panel width) instead of a narrow side-by-side flex row.
#   2. Duplicated "upload upload" text in file-uploader dropzones -> the
#      previous blanket `[data-testid="stFileUploaderDropzone"] *` selector
#      (which could unintentionally affect elements Streamlit keeps visually
#      deduplicated) has been narrowed to specific, known sub-elements.
# ---------------------------------------------------------------------------
_CSS = """
<style>
:root {
  --bg: #0A0A0C;
  --panel: #111114;
  --panel-alt: #17171B;
  --border: #262629;
  --border-strong: #3D3D42;
  --text: #E8E8E6;
  --text-dim: #93939A;
  --text-faint: #66666C;
  --accent: #C98A3A;
  --accent-strong: #E0A253;
  --accent2: #4FA8C9;
  --danger: #D6493F;

  --font-sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --font-mono: ui-monospace, SFMono-Regular, Consolas, 'Liberation Mono', monospace;

  --fs-display: 1.65rem;
  --fs-title: 1.02rem;
  --fs-section: 0.9rem;
  --fs-body: 1.02rem;
  --fs-label: 0.68rem;
  --fs-metadata: 0.78rem;
  --fs-mono: 0.76rem;

  --sp-1: 4px; --sp-2: 8px; --sp-3: 12px; --sp-4: 16px; --sp-6: 24px; --sp-8: 32px; --sp-12: 48px;

  --radius-outer: 4px;
  --radius-inner: 2px;
}

.stApp { background: var(--bg); color: var(--text); font-family: var(--font-sans); }
.main .block-container { max-width: 1280px; padding-top: var(--sp-6); }
h1, h2, h3, h4, p, span, label, div { font-family: var(--font-sans); color: var(--text); }

/* --- compact control-rail sidebar --- */
section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"][aria-expanded="true"] { min-width: 250px !important; max-width: 300px !important; }
section[data-testid="stSidebar"] .block-container { padding-top: var(--sp-6); }

/* --- segmented navigation: filled pill for the active tab, not just an
   underline, so the current position reads at a glance --- */
.stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid var(--border); gap: 2px; margin-bottom: var(--sp-4); }
.stTabs [data-baseweb="tab"] {
  background: transparent; border: none; border-bottom: 2px solid transparent;
  color: var(--text-dim); font-family: var(--font-sans); font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.04em; font-size: var(--fs-label); border-radius: var(--radius-inner) var(--radius-inner) 0 0 !important;
  padding: var(--sp-3) var(--sp-4); transition: background 150ms ease, color 150ms ease;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text); background: var(--panel-alt); }
.stTabs [aria-selected="true"] { color: var(--text) !important; border-bottom-color: var(--accent) !important; background: var(--panel-alt) !important; }

/* --- buttons / inputs. Streamlit's default button kind is "secondary" -
   used here for informational / suggestion controls (cool accent2 outline).
   Execution actions (Run analysis, Ask a follow-up) explicitly pass
   type="primary" and get the amber fill instead - see color-role comment
   in docs/ui_design.md Revision 4. --- */
/* min-width: 0 lets the ellipsis rule below actually take effect when a
   button sits inside a flex row (e.g. the example-query chips, each in its
   own st.columns() cell) - flex items default to min-width: auto, which
   otherwise refuses to shrink below the untruncated text's width even with
   overflow: hidden on the child. Found via an actual rendered screenshot
   of this exact CSS, not by inspection alone - see docs/ui_design.md. */
.stButton { min-width: 0; }
.stButton > button {
  border-radius: var(--radius-inner); font-family: var(--font-sans); font-weight: 600;
  font-size: var(--fs-metadata); letter-spacing: 0.01em; padding: var(--sp-2) var(--sp-4); width: 100%;
  transition: background 150ms ease, color 150ms ease, border-color 150ms ease, transform 80ms ease;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.stButton > button:active { transform: scale(0.98); }
.stButton > button:disabled { background: var(--panel-alt) !important; color: var(--text-faint) !important; border-color: var(--border) !important; }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible { outline: 2px solid var(--accent2); outline-offset: 2px; }

.stButton > button[kind="primary"] { background: var(--accent); color: #17110A; border: 1px solid var(--accent); }
.stButton > button[kind="primary"]:hover { background: var(--accent-strong); border-color: var(--accent-strong); }

.stButton > button[kind="secondary"] { background: transparent; color: var(--accent2); border: 1px solid var(--accent2); font-weight: 500; }
.stButton > button[kind="secondary"]:hover { background: rgba(79, 168, 201, 0.12); }

.stDownloadButton > button {
  border-radius: var(--radius-inner); font-family: var(--font-sans); font-size: var(--fs-metadata);
  transition: border-color 150ms ease, color 150ms ease, background 150ms ease;
}
.stDownloadButton > button[kind="primary"] { background: var(--accent); color: #17110A; border: 1px solid var(--accent); font-weight: 700; }
.stDownloadButton > button[kind="primary"]:hover { background: var(--accent-strong); border-color: var(--accent-strong); }
.stDownloadButton > button[kind="secondary"] { background: transparent; color: var(--text); border: 1px solid var(--border-strong); font-weight: 500; }
.stDownloadButton > button[kind="secondary"]:hover { border-color: var(--accent2); color: var(--accent2); }

.stTextInput input, .stDateInput input, .stSelectbox div[data-baseweb="select"] > div {
  background: var(--panel-alt) !important; color: var(--text) !important; border: 1px solid var(--border) !important;
  border-radius: 3px !important; font-family: var(--font-sans) !important; font-size: var(--fs-metadata) !important;
}
.stTextInput input:focus, .stDateInput input:focus { border-color: var(--accent) !important; }

/* File uploader - narrowed selectors (see module docstring above the CSS
   block for why the previous universal-selector approach is suspected to
   have caused duplicated placeholder text). */
[data-testid="stFileUploaderDropzone"] { background: var(--panel-alt); border: 1px dashed var(--border-strong) !important; border-radius: 3px !important; }
[data-testid="stFileUploaderDropzone"] section { color: var(--text-dim); font-family: var(--font-sans); }
[data-testid="stFileUploaderDropzoneInstructions"] { color: var(--text-dim) !important; font-family: var(--font-sans) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] span { font-family: var(--font-sans) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] small { color: var(--text-faint) !important; font-family: var(--font-sans) !important; }
[data-testid="stFileUploaderFile"] { color: var(--text-dim); font-family: var(--font-sans); }
[data-testid="baseButton-secondary"] { font-family: var(--font-sans) !important; }

.stCheckbox label p { font-size: var(--fs-metadata); color: var(--text-dim); }

[data-testid="stExpander"] { background: var(--panel); border: 1px solid var(--border) !important; border-radius: 3px !important; margin-bottom: var(--sp-2); }
[data-testid="stExpander"] summary { font-family: var(--font-sans); font-weight: 600; font-size: var(--fs-metadata); color: var(--text-dim); }
[data-testid="stExpander"] summary:hover { color: var(--text); }

[data-testid="stAlert"] { border-radius: 3px !important; font-family: var(--font-sans); font-size: var(--fs-metadata); }
[data-testid="stCaptionContainer"] { font-size: var(--fs-metadata); color: var(--text-faint); }
hr { border-color: var(--border); margin: var(--sp-3) 0; }

/* --- structural components --- */
.sq-header { border-bottom: 1px solid var(--border); padding-bottom: var(--sp-4); margin-bottom: var(--sp-2); }
.sq-header-top { display: flex; align-items: flex-start; justify-content: space-between; gap: var(--sp-4); flex-wrap: wrap; }
.sq-display { font-family: var(--font-sans); font-size: var(--fs-display); font-weight: 700; color: var(--text); letter-spacing: -0.02em; }
.sq-header-section { font-family: var(--font-sans); font-size: var(--fs-section); font-weight: 500; color: var(--text-dim); margin-top: 2px; }
.sq-header-meta { font-family: var(--font-sans); font-size: var(--fs-metadata); color: var(--text-faint); margin-top: 3px; }
.sq-status-badge { display: inline-flex; align-items: center; gap: 6px; font-family: var(--font-sans); font-size: var(--fs-label); font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; padding: 5px 10px; border: 1px solid var(--border-strong); border-radius: var(--radius-inner); color: var(--text-dim); white-space: nowrap; }
.sq-status-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--text-faint); flex-shrink: 0; }
.sq-status-ready { color: var(--accent2); border-color: var(--accent2); }
.sq-status-ready .sq-status-dot { background: var(--accent2); }

.sq-rail-title { font-family: var(--font-sans); font-size: var(--fs-label); font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-faint); margin: var(--sp-4) 0 var(--sp-2) 0; }
.sq-rail-row { display: flex; align-items: center; gap: var(--sp-2); font-size: var(--fs-metadata); padding: 3px 0; }
.sq-rail-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent2); flex-shrink: 0; }
.sq-rail-dot.off { background: var(--text-faint); }
.sq-rail-label { color: var(--text); flex: 1; }
.sq-rail-state { color: var(--text-dim); font-size: var(--fs-label); letter-spacing: 0.04em; font-weight: 600; }

/* Registry rows - stacked (label above, value below), full panel width, so
   a long tool id gets the whole sidebar's width to wrap in rather than a
   narrow half-column. This is the concrete fix for the reported
   character-by-character wrapping bug. */
.sq-reg-row { padding: var(--sp-2) 0; border-bottom: 1px solid var(--border); }
.sq-reg-row:last-child { border-bottom: none; }
.sq-reg-name { font-family: var(--font-mono); font-size: 0.66rem; line-height: 1.35; color: var(--text-dim); overflow-wrap: anywhere; }
.sq-reg-status { font-family: var(--font-sans); font-size: var(--fs-label); font-weight: 700; letter-spacing: 0.05em; margin-top: 2px; }

.sq-nav-eyebrow { font-family: var(--font-sans); font-size: var(--fs-label); font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-faint); margin-bottom: 2px; }
.sq-nav-descriptor { font-family: var(--font-sans); font-size: var(--fs-title); font-weight: 650; color: var(--text); margin: 0 0 var(--sp-3) 0; }
.sq-analysis-type { font-family: var(--font-sans); font-size: var(--fs-metadata); color: var(--text-dim); margin-bottom: var(--sp-2); }
.sq-run-meta { font-family: var(--font-mono); font-size: var(--fs-label); color: var(--accent2); letter-spacing: 0.04em; margin-bottom: var(--sp-2); }

.sq-panel { border: 1px solid var(--border); border-top-color: var(--border-strong); background: var(--panel); border-radius: var(--radius-outer); padding: var(--sp-3) var(--sp-4); margin-bottom: var(--sp-3); }
.sq-panel-danger { border-color: var(--danger); }
.sq-panel-label { font-family: var(--font-sans); font-size: var(--fs-label); font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--text-faint); margin-bottom: var(--sp-2); }

.sq-kv { display: flex; gap: var(--sp-3); font-size: var(--fs-metadata); padding: 3px 0; }
.sq-k { color: var(--text-dim); min-width: 90px; flex-shrink: 0; }
.sq-v { color: var(--text); word-break: break-word; }
.sq-v.mono { font-family: var(--font-mono); font-size: var(--fs-mono); }

.sq-answer { font-family: var(--font-sans); font-size: var(--fs-body); font-weight: 450; line-height: 1.6; color: var(--text); padding: var(--sp-1) 0 var(--sp-3) 0; }
.sq-clean-state { font-family: var(--font-sans); font-size: 1.1rem; font-weight: 700; color: var(--accent2); padding: var(--sp-1) 0 var(--sp-1) 0; letter-spacing: -0.01em; }

.sq-conf-number { font-family: var(--font-sans); font-size: 1.6rem; font-weight: 700; color: var(--accent); font-variant-numeric: tabular-nums; line-height: 1; }
.sq-conf-label { font-family: var(--font-sans); font-size: var(--fs-metadata); color: var(--text-dim); margin-top: 2px; }
.sq-conf-caveat { font-family: var(--font-sans); font-size: var(--fs-label); color: var(--text-faint); font-style: italic; margin-top: 2px; }
.sq-conf-bar-track { height: 4px; background: var(--panel-alt); border-radius: 2px; margin-top: var(--sp-2); overflow: hidden; }
.sq-conf-bar-fill { height: 100%; background: var(--accent); }
.sq-downgrade { font-family: var(--font-sans); font-size: var(--fs-label); color: var(--danger); padding-top: var(--sp-1); }

.sq-danger-banner { border: 1px solid var(--danger); border-radius: var(--radius-outer); color: var(--text); background: rgba(214,73,63,0.08); font-family: var(--font-sans); font-size: var(--fs-metadata); padding: var(--sp-2) var(--sp-3); margin-bottom: var(--sp-3); }
.sq-danger-banner b { color: var(--danger); }

.sq-sample-note { font-family: var(--font-sans); font-size: var(--fs-label); color: var(--text-faint); font-style: italic; margin: -2px 0 var(--sp-2) 0; }
.sq-evidence-caption { font-family: var(--font-sans); font-size: var(--fs-metadata); font-weight: 500; color: var(--text-dim); margin: var(--sp-3) 0 var(--sp-2) 0; }
[data-testid="stImage"] img { border: 1px solid var(--border-strong); border-radius: var(--radius-inner); }
.sq-trace-flow { font-family: var(--font-sans); font-size: var(--fs-metadata); color: var(--text); line-height: 1.9; }
.sq-trace-flow .arrow { color: var(--text-faint); margin: 0 var(--sp-1); }
.sq-trace-flow .step-label { color: var(--text-faint); font-weight: 600; text-transform: uppercase; font-size: var(--fs-label); letter-spacing: 0.04em; margin-right: var(--sp-2); }
.sq-trace-reasoning { font-family: var(--font-sans); font-size: var(--fs-metadata); color: var(--text-dim); margin: var(--sp-1) 0 var(--sp-2) var(--sp-6); }

/* --- agent process timeline: a real vertical connector (dot + line), not
   a flat text accordion - status still comes ONLY from ExecutionTrace via
   routing/timeline.py; there is no "currently running" visual because a
   trace only renders once the run has already finished. --- */
.sq-tl { display: flex; flex-direction: column; }
.sq-tl-item { display: flex; gap: var(--sp-3); }
.sq-tl-marker-col { display: flex; flex-direction: column; align-items: center; width: 12px; flex-shrink: 0; }
.sq-tl-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; margin-top: 3px; }
.sq-tl-dot.sq-tl-done { background: var(--accent2); }
.sq-tl-dot.sq-tl-failed { background: var(--danger); }
.sq-tl-dot.sq-tl-pending { background: var(--text-faint); }
.sq-tl-connector { width: 1px; flex: 1; min-height: 12px; background: var(--border-strong); margin: 2px 0; }
.sq-tl-body { padding-bottom: var(--sp-3); }
.sq-tl-label { font-family: var(--font-sans); font-size: var(--fs-metadata); color: var(--text); font-weight: 500; }
.sq-tl-detail { font-family: var(--font-sans); font-size: var(--fs-label); color: var(--text-dim); margin-top: 1px; }

/* --- input guidance / validation / example chips --- */
.sq-validation-note { font-family: var(--font-sans); font-size: var(--fs-label); color: var(--text-faint); margin: var(--sp-1) 0 var(--sp-2) 0; }
.sq-validation-note b { color: var(--text-dim); letter-spacing: 0.04em; }
.sq-chip-label { font-family: var(--font-sans); font-size: var(--fs-label); color: var(--text-faint); text-transform: uppercase; letter-spacing: 0.06em; margin: var(--sp-3) 0 4px 0; }
.sq-ask-label { font-family: var(--font-sans); font-size: var(--fs-label); color: var(--text-faint); text-transform: uppercase; letter-spacing: 0.06em; margin: var(--sp-4) 0 4px 0; }

/* --- conversation / analysis session transcript --- */
.sq-convo-turn { font-family: var(--font-sans); font-size: var(--fs-metadata); padding: 3px 0; color: var(--text); }
.sq-convo-answer { color: var(--text-dim); }
.sq-convo-who { font-weight: 700; font-size: var(--fs-label); text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-faint); margin-right: var(--sp-2); }

/* --- error state (400/500) --- */
.sq-error-eyebrow { font-family: var(--font-sans); font-size: var(--fs-label); font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--danger); margin-bottom: 4px; }
.sq-error-title { font-family: var(--font-sans); font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: var(--sp-2); letter-spacing: -0.01em; }
.sq-error-why { font-family: var(--font-sans); font-size: var(--fs-metadata); color: var(--text-dim); margin-bottom: var(--sp-3); }
.sq-error-label { font-family: var(--font-sans); font-size: var(--fs-label); font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--text-faint); margin-bottom: 4px; }
.sq-error-next { font-family: var(--font-sans); font-size: var(--fs-metadata); color: var(--text); }

/* --- empty state --- */
.sq-empty { font-family: var(--font-sans); border: 1px solid var(--border); border-radius: var(--radius-outer); padding: var(--sp-6) var(--sp-4); letter-spacing: 0.02em; }
.sq-empty-title { font-family: var(--font-sans); font-size: 1rem; font-weight: 700; color: var(--text); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
.sq-empty-sub { font-family: var(--font-sans); font-size: var(--fs-metadata); color: var(--text-dim); margin-bottom: var(--sp-4); }
.sq-empty-choices { display: flex; gap: var(--sp-2); flex-wrap: wrap; }
.sq-empty-choice { font-family: var(--font-sans); font-size: var(--fs-label); color: var(--text-faint); border: 1px solid var(--border); border-radius: var(--radius-inner); padding: 4px 10px; letter-spacing: 0.04em; }
.sq-empty-choice-current { color: var(--accent2); border-color: var(--accent2); }

/* --- AERIS AI pivot: mission cards, mission brief, analysis cards, sensor
   readiness badge. Additive only - reuses the existing token palette
   (--panel/--border/--accent/--accent2/--text-dim etc.), no new colors, no
   layout system change. --- */
.aeris-eyebrow { font-family: var(--font-sans); font-size: var(--fs-label); font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-faint); margin: var(--sp-2) 0 4px 0; }
.aeris-subtitle { font-family: var(--font-sans); font-size: var(--fs-metadata); color: var(--text-dim); margin-bottom: var(--sp-3); }

.aeris-mission-row { display: flex; flex-direction: column; }
.aeris-mission-card { text-align: left !important; }
.aeris-mission-desc { font-family: var(--font-sans); font-size: var(--fs-label); color: var(--text-faint); margin: -2px 0 var(--sp-2) 0; }

.aeris-analysis-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: var(--sp-2); margin-bottom: var(--sp-3); }
@media (max-width: 900px) { .aeris-analysis-grid { grid-template-columns: repeat(2, 1fr); } }
.aeris-analysis-card { border: 1px solid var(--border); border-radius: var(--radius-outer); background: var(--panel); padding: var(--sp-2) var(--sp-3); }
.aeris-analysis-name { font-family: var(--font-sans); font-size: var(--fs-metadata); font-weight: 700; color: var(--text); }
.aeris-analysis-desc { font-family: var(--font-sans); font-size: var(--fs-label); color: var(--text-dim); margin-top: 2px; min-height: 28px; }
.aeris-analysis-backend { font-family: var(--font-sans); font-size: var(--fs-label); font-weight: 700; letter-spacing: 0.05em; margin-top: var(--sp-2); }

.aeris-brief { border: 1px solid var(--accent2); border-radius: var(--radius-outer); background: var(--panel); padding: var(--sp-3) var(--sp-4); margin-bottom: var(--sp-3); }
.aeris-brief-title { font-family: var(--font-sans); font-size: var(--fs-label); font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent2); margin-bottom: var(--sp-2); }
.aeris-brief-finding { font-family: var(--font-sans); font-size: var(--fs-body); font-weight: 500; color: var(--text); line-height: 1.5; margin-bottom: var(--sp-2); }
.aeris-brief-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: var(--sp-2) var(--sp-4); }
.aeris-brief-k { font-family: var(--font-sans); font-size: var(--fs-label); color: var(--text-faint); text-transform: uppercase; letter-spacing: 0.04em; }
.aeris-brief-v { font-family: var(--font-sans); font-size: var(--fs-metadata); color: var(--text); margin-top: 1px; }

.aeris-sensor-ready { font-family: var(--font-mono); font-size: var(--fs-metadata); color: var(--accent2); line-height: 1.6; margin-top: var(--sp-2); white-space: pre; }

.aeris-orch-wrap { border: 1px solid var(--border-strong); border-radius: var(--radius-outer); background: var(--panel); padding: var(--sp-3) var(--sp-4); margin: var(--sp-2) 0 var(--sp-3) 0; }
.aeris-orch-title { font-family: var(--font-sans); font-size: var(--fs-label); font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-faint); margin-bottom: var(--sp-2); }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Shared helpers - used by every tab, so there is one rendering path per
# concept regardless of which capability produced the ExecutionTrace.
# ---------------------------------------------------------------------------

def _tool_available(name: str) -> bool:
    try:
        return registry.get(name).status == registry.AVAILABLE
    except KeyError:
        return False


def _capability_ready(tool_names) -> bool:
    return any(_tool_available(n) for n in tool_names)


def _capability_backend_label(tool_names) -> Tuple[str, bool]:
    """Distinguishes a genuinely available real pretrained model from a
    classical-CV baseline from a capability that is unavailable altogether -
    all three states read live off the registry, never inferred from
    tool_name string matching or hardcoded per-capability assumptions
    (today only the VQA tool_name equal to SMOLVLM_TOOL_NAME is ever a real
    pretrained model; every other registered tool is a classical baseline)."""
    for name in tool_names:
        if _tool_available(name):
            return ("REAL MODEL", True) if name == SMOLVLM_TOOL_NAME else ("CLASSICAL", True)
    return ("UNAVAILABLE", False)


def _active_label(prefix: str) -> str:
    """What actually, really executed most recently for this capability in
    THIS session - '—' if nothing has been run yet. Never lit just
    because a tab was opened; only a real ExecutionTrace with a real
    SpecialistOutput sets this."""
    trace = st.session_state.get(f"{prefix}_trace")
    if trace is None or trace.specialist_output is None:
        return "—"
    out = trace.specialist_output
    return out.model_name if out.model_name else "Classical"


def _truncate(text: Optional[str], n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[: max(0, n - 1)].rstrip() + "…"


def _new_run_dir(tab_key: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    d = os.path.join(RUNS_DIR, f"{tab_key}_{ts}")
    os.makedirs(d, exist_ok=True)
    return d


def _save_upload(uploaded_file, dest_dir: str, stem: str) -> str:
    ext = os.path.splitext(uploaded_file.name)[1] or ".png"
    path = os.path.join(dest_dir, f"{stem}{ext}")
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return path


def _set_session_value(key: str, value: Any) -> None:
    st.session_state[key] = value


def _seeded_text_input(label: str, key: str, default: str, **kwargs) -> str:
    """A key-bound st.text_input() whose STARTING value is `default`,
    without ever passing value= to the widget itself. Streamlit forbids
    combining value= with a key that may also be set via the Session
    State API - which render_example_chips's suggestion-chip on_click
    (_set_session_value) does for this same key - and warns: "created
    with a default value but also had its value set via the Session
    State API." Seeding st.session_state BEFORE the widget is created has
    the same visible effect (the box starts pre-filled with `default`)
    without that conflict, and leaves typed input, chip clicks, reruns,
    and follow-ups all working exactly as before."""
    st.session_state.setdefault(key, default)
    return st.text_input(label, key=key, **kwargs)


def _trigger_rerun() -> None:
    """Restarts script execution from the top so a follow-up query's freshly
    stored result renders immediately instead of waiting for the next
    unrelated interaction. No-ops harmlessly when neither rerun API exists
    (e.g. under this project's fake-streamlit test stub, which intentionally
    does not implement rerun - see tests/integration/test_streamlit_app_logic.py)."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def _record_run(prefix: str, trace: ExecutionTrace, run_dir: str, query: str, extra=None) -> None:
    """Stores a freshly executed run as "current" for this tab, pushing
    whatever was previously current into that tab's real conversation
    history first - used identically for a tab's first run and for every
    follow-up question, so history always reflects real prior queries and
    real prior traces, never placeholder text."""
    if f"{prefix}_trace" in st.session_state:
        history_key = f"{prefix}_history"
        st.session_state.setdefault(history_key, [])
        st.session_state[history_key].append(
            (st.session_state.get(f"{prefix}_query_used", ""), st.session_state[f"{prefix}_trace"])
        )
    st.session_state[f"{prefix}_trace"] = trace
    st.session_state[f"{prefix}_rundir"] = run_dir
    st.session_state[f"{prefix}_query_used"] = query
    if extra is not None:
        st.session_state[f"{prefix}_extra"] = extra
    st.session_state["_run_count"] = st.session_state.get("_run_count", 0) + 1
    st.session_state["_last_query"] = query


def _execute_and_record(prefix: str, mode: str, inputs: Dict[str, Any], query: str, run_dir: Optional[str] = None) -> None:
    """The one real execution path every tab's initial run AND every
    follow-up question both call - always the same `pipeline.run_query()`,
    never a UI-specific variant. `inputs` is the small, compact structured
    session context (image path(s), modality, dates) cached from the
    original run; a follow-up sends only the new query text alongside it,
    never a raw conversation transcript, to the specialist. `run_dir`, when
    given, is the directory an initial run already created to save uploaded
    file(s) into - reused as the output directory too, so one run produces
    one run directory, not two; a follow-up (no new upload to save) omits it
    and gets a fresh one.

    `mode="auto"` is used by the unified "Ask AERIS" tab, which (by
    design - see docs/ui_design.md / the router's own paired-image
    disambiguation rules) does not know in advance which of the four real
    task_types the router will choose - unlike the four fixed-purpose tabs,
    which always pass their own real mode name because a fixed-purpose tab's
    mode IS the task type it dispatches. After the real `trace` comes back,
    "auto" is resolved from the REAL `trace.router_decision.task_type` the
    router actually produced, never guessed ahead of time - this is a
    read-only relabeling for the extra-evidence panel captions, not a
    second routing decision."""
    if run_dir is None:
        run_dir = _new_run_dir(prefix)
    kwargs = dict(inputs)
    kwargs["query"] = query
    kwargs["out_dir"] = run_dir
    with st.spinner("Running analysis - this can take under a minute on CPU..."):
        trace = pipeline.run_query(**kwargs)

    if mode == "auto":
        real_task_type = trace.router_decision.task_type if trace.router_decision else None
        mode = {
            "optical_sar_fusion": "fusion",
            "bitemporal_change": "change",
        }.get(real_task_type)

    extra = None
    if mode == "fusion":
        raw_a = raster_io.load(inputs["image1_path"]).array
        raw_b = raster_io.load(inputs["image2_path"]).array
        extra = [
            ("Optical (input)", composer.to_displayable_rgb(raw_a)),
            ("SAR (input)", composer.to_displayable_rgb(raw_b)),
        ]
    elif mode == "change":
        raw_a = raster_io.load(inputs["image1_path"]).array
        raw_b = raster_io.load(inputs["image2_path"]).array
        extra = [
            ("Before", composer.to_displayable_rgb(raw_a)),
            ("After", composer.to_displayable_rgb(raw_b)),
        ]

    st.session_state[f"{prefix}_inputs"] = inputs
    _record_run(prefix, trace, run_dir, query, extra=extra)


def _system_status() -> Tuple[int, int]:
    """How many of the four real capabilities are actually ready right now,
    read live off the registry (the exact same tool-availability check the
    sidebar's "System" rail already uses) - never a hardcoded claim."""
    capability_tools = [
        [SMOLVLM_TOOL_NAME, CLASSICAL_VQA_TOOL_NAME],
        ["tool_grounding_v0"],
        ["tool_change_v0"],
        ["tool_fusion_v0"],
    ]
    ready = sum(1 for tools in capability_tools if _capability_ready(tools))
    return ready, len(capability_tools)


def render_header() -> None:
    ready, total = _system_status()
    if ready >= total:
        badge = '<span class="sq-status-badge sq-status-ready"><span class="sq-status-dot"></span>SYSTEM READY</span>'
    else:
        badge = (
            '<span class="sq-status-badge"><span class="sq-status-dot"></span>'
            f"PARTIAL SYSTEM &middot; {ready}/{total} READY</span>"
        )
    st.markdown(
        f"""
        <div class="sq-header">
          <div class="sq-header-top">
            <div>
              <div class="sq-display">AERIS AI</div>
              <div class="sq-header-section">Agentic Earth Observation Intelligence</div>
              <div class="sq-header-meta">Student Innovation &middot; Space Technology &middot; Software &middot; SIH26209</div>
            </div>
            {badge}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="sq-rail-title">System</div>', unsafe_allow_html=True)
        capabilities = [
            ("VQA", [SMOLVLM_TOOL_NAME, CLASSICAL_VQA_TOOL_NAME]),
            ("Ground", ["tool_grounding_v0"]),
            ("Change", ["tool_change_v0"]),
            ("Fusion", ["tool_fusion_v0"]),
        ]
        for label, tool_names in capabilities:
            state_label, ready = _capability_backend_label(tool_names)
            dot_class = "sq-rail-dot" if ready else "sq-rail-dot off"
            color = _BACKEND_STATE_COLOR.get(state_label, "var(--text-dim)")
            st.markdown(
                f'<div class="sq-rail-row"><span class="{dot_class}"></span>'
                f'<span class="sq-rail-label">{html.escape(label)}</span>'
                f'<span class="sq-rail-state" style="color:{color}">{html.escape(state_label)}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sq-rail-title">Active</div>', unsafe_allow_html=True)
        for label, prefix in (("VQA", "vqa"), ("Ground", "gr"), ("Change", "ch"), ("Fusion", "fu")):
            st.markdown(
                f'<div class="sq-rail-row"><span class="sq-rail-label">{html.escape(label)}</span>'
                f'<span class="sq-rail-state">{html.escape(_active_label(prefix))}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sq-rail-title">Session</div>', unsafe_allow_html=True)
        run_count = st.session_state.get("_run_count", 0)
        st.markdown(
            f'<div class="sq-rail-row"><span class="sq-rail-label">Analyses run</span>'
            f'<span class="sq-rail-state">{run_count}</span></div>',
            unsafe_allow_html=True,
        )
        last_query = st.session_state.get("_last_query")
        if last_query:
            st.caption(f"Last query: {_truncate(last_query, 60)}")

        with st.expander("Technical system status"):
            st.caption(f"Build: {BUILD_LABEL}")
            for name, entry in sorted(registry.all_entries().items()):
                color = "var(--accent2)" if entry.status == registry.AVAILABLE else "var(--text-faint)"
                st.markdown(
                    f'<div class="sq-reg-row"><div class="sq-reg-name">{html.escape(name)}</div>'
                    f'<div class="sq-reg-status" style="color:{color}">{html.escape(entry.status)}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="sq-rail-title">About</div>', unsafe_allow_html=True)
        st.caption(
            "Grounding, bi-temporal change, and optical+SAR fusion use classical "
            "computer-vision baselines, not trained foundation models."
        )
        st.caption("Developed as SatQuery AI (SIH26167 dev history); submitted as AERIS AI.")


_AERIS_ANALYSIS_CARDS = (
    ("VQA", "Vision-language scene understanding", [SMOLVLM_TOOL_NAME, CLASSICAL_VQA_TOOL_NAME]),
    ("GROUNDING", "Text-guided spatial localisation", ["tool_grounding_v0"]),
    ("CHANGE", "Bi-temporal change analysis", ["tool_change_v0"]),
    ("OPTICAL + SAR", "Cross-modal Earth-observation analysis", ["tool_fusion_v0"]),
)


def render_aeris_analysis_cards() -> None:
    """Phase 7 "Analysis card" per specialist. Every value here is the exact
    same live registry-derived (REAL MODEL / CLASSICAL / UNAVAILABLE) status
    `_capability_backend_label` already computes for the sidebar "System"
    rail - this is a second, more visible surface for the same real data,
    not a new claim. Explicitly discloses learned-model vs. classical-CV
    baseline per card, never hidden, per the pivot brief's "transparency is
    part of the product" instruction."""
    cols = st.columns(len(_AERIS_ANALYSIS_CARDS))
    for col, (name, desc, tool_names) in zip(cols, _AERIS_ANALYSIS_CARDS):
        state_label, _ready = _capability_backend_label(tool_names)
        color = _BACKEND_STATE_COLOR.get(state_label, "var(--text-dim)")
        with col:
            st.markdown(
                f'<div class="aeris-analysis-card">'
                f'<div class="aeris-analysis-name">{html.escape(name)}</div>'
                f'<div class="aeris-analysis-desc">{html.escape(desc)}</div>'
                f'<div class="aeris-analysis-backend" style="color:{color}">{html.escape(state_label)}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


def render_mode_help(mode_key: str) -> None:
    info = MODE_HELP[mode_key]
    st.caption(info["provide"])
    with st.expander("About this analysis mode"):
        st.markdown(
            f'<div class="sq-kv"><span class="sq-k">Why</span><span class="sq-v">{html.escape(info["why"])}</span></div>'
            f'<div class="sq-kv"><span class="sq-k">Format</span><span class="sq-v">{html.escape(info["format"])}</span></div>'
            f'<div class="sq-kv"><span class="sq-k">Then</span><span class="sq-v">{html.escape(info["happens"])}</span></div>',
            unsafe_allow_html=True,
        )


def render_example_chips(mode_key: str, query_key: str) -> None:
    """Non-executing starter prompts. Rendered with Streamlit's default
    ("secondary") button kind - the cool muted accent2 outline, visually
    distinct from the amber "Run analysis" / "Ask" execution actions - so it
    is never mistaken for a control that runs analysis (see Section 7/8,
    docs/ui_design.md Revision 4)."""
    suggestions = EXAMPLE_QUERIES.get(mode_key, [])
    if not suggestions:
        return
    st.markdown('<div class="sq-chip-label">Suggested queries</div>', unsafe_allow_html=True)
    cols = st.columns(len(suggestions))
    for i, (col, suggestion) in enumerate(zip(cols, suggestions)):
        with col:
            st.button(
                _truncate(suggestion, 34),
                key=f"{query_key}_chip_{i}",
                on_click=_set_session_value,
                args=(query_key, suggestion),
                type="secondary",
            )


# Phase 3 "Mission" presets: UX-only shortcuts, same on_click/session_state
# mechanism as render_example_chips above (`_set_session_value`) - a click
# only populates the query text field with a representative query for that
# application area. These are NOT domain-specific trained models: every
# mission still runs through the exact same router.decide() and the exact
# same four real specialists every other query does; the card is a starting
# point, not a claim about a specialized model existing for that domain.
AERIS_MISSIONS: List[Tuple[str, str, str]] = [
    ("AGRICULTURE", "Land-cover and vegetation analysis", "Describe the major land cover types visible in this image."),
    ("URBAN", "Built-up and infrastructure analysis", "Highlight the urban / built-up region."),
    ("WATER", "Water-region identification and monitoring", "Locate the water body in this image and highlight it."),
    ("CHANGE", "Bi-temporal change investigation", "What changed between these two dates?"),
    ("MULTIMODAL", "Optical + SAR joint analysis", "Use both images to identify built-up and water-covered regions."),
]


def render_mission_selector(query_key: str) -> None:
    st.markdown('<div class="sq-chip-label">Mission</div>', unsafe_allow_html=True)
    cols = st.columns(len(AERIS_MISSIONS))
    for i, (col, (name, desc, suggested_query)) in enumerate(zip(cols, AERIS_MISSIONS)):
        with col:
            st.button(
                name,
                key=f"{query_key}_mission_{i}",
                on_click=_set_session_value,
                args=(query_key, suggested_query),
                type="secondary",
                help=desc,
            )
            st.markdown(f'<div class="aeris-mission-desc">{html.escape(desc)}</div>', unsafe_allow_html=True)


def render_section_header(prefix: str, kind: str) -> None:
    """Replaces the old bare "Input"/"Result" text with the redesigned
    information architecture: the input column gets this capability's real
    static navigation copy (number, title, one-line descriptor - the same
    TAB_INFO already used for the empty state), the result column gets a
    plain eyebrow since its real headline is the analysis type/answer
    rendered just below it, not a repeated label."""
    if kind == "input":
        num, title, descriptor = TAB_INFO[prefix]
        st.markdown(
            f'<div class="sq-nav-eyebrow">{html.escape(num)} &middot; INPUT</div>'
            f'<div class="sq-nav-descriptor">{html.escape(title)} — {html.escape(descriptor)}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="sq-nav-eyebrow">RESULT</div>', unsafe_allow_html=True)


def render_validation_note(note: Optional[Tuple[str, str]]) -> None:
    if note is None:
        return
    category, message = note
    st.markdown(
        f'<div class="sq-validation-note"><b>{html.escape(category)}</b> &middot; {html.escape(message)}</div>',
        unsafe_allow_html=True,
    )


def render_analysis_type(decision) -> None:
    label = TASK_TYPE_LABELS.get(decision.task_type, decision.task_type)
    st.markdown(f'<div class="sq-analysis-type">Analysis type: {html.escape(label)}</div>', unsafe_allow_html=True)


def render_run_meta(trace: ExecutionTrace) -> None:
    ts = datetime.datetime.fromtimestamp(trace.started_at).strftime("%H:%M:%S")
    badges = [f"RUN {ts}"]
    for s in trace.input_summary:
        modality = s.get("modality") if isinstance(s, dict) else None
        if modality:
            badges.append(str(modality).upper())
    st.markdown(
        '<div class="sq-run-meta">' + " &middot; ".join(html.escape(b) for b in badges) + "</div>",
        unsafe_allow_html=True,
    )


def render_validation_notes(trace: ExecutionTrace) -> None:
    warnings = trace.validation.get("warnings") if trace.validation else None
    if not warnings:
        return
    with st.expander(f"Validation notes ({len(warnings)})"):
        for w in warnings:
            st.markdown(f"- {html.escape(w)}")


def render_confidence_compact(conf) -> None:
    pct = max(0.0, min(1.0, conf.value)) * 100
    short_label = CONFIDENCE_SHORT_LABEL.get(conf.method_version, conf.method_version)
    caveat = CONFIDENCE_CAVEAT.get(conf.method_version, "")
    downgrades_html = "".join(f'<div class="sq-downgrade">&#8226; {html.escape(d)}</div>' for d in conf.downgrades)
    st.markdown(
        f"""
        <div class="sq-panel">
          <div class="sq-panel-label">Confidence</div>
          <div class="sq-conf-number">{conf.value:.2f}</div>
          <div class="sq-conf-label">{html.escape(short_label)}</div>
          <div class="sq-conf-caveat">{html.escape(caveat)}</div>
          <div class="sq-conf-bar-track"><div class="sq-conf-bar-fill" style="width:{pct:.1f}%"></div></div>
          {downgrades_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Confidence details"):
        st.markdown(
            f'<div class="sq-kv"><span class="sq-k">Method</span><span class="sq-v mono">{html.escape(conf.method_version)}</span></div>'
            f'<div class="sq-kv"><span class="sq-k">Measures</span><span class="sq-v">{html.escape(conf.basis_description)}</span></div>',
            unsafe_allow_html=True,
        )


def render_active_specialist(out: SpecialistOutput) -> None:
    role = TASK_TYPE_LABELS.get(out.task_type, out.task_type)
    if out.model_name:
        backend = out.raw.get("backend", "CPU") if isinstance(out.raw, dict) else "CPU"
        rows = [("Model", out.model_name), ("Role", role), ("Backend", str(backend)), ("Latency", f"{out.latency_seconds:.1f}s")]
    else:
        rows = [("Method", "Classical CV baseline"), ("Role", role), ("Latency", f"{out.latency_seconds:.1f}s")]

    kv_html = "".join(
        f'<div class="sq-kv"><span class="sq-k">{html.escape(k)}</span><span class="sq-v">{html.escape(str(v))}</span></div>'
        for k, v in rows
    )
    st.markdown(f'<div class="sq-panel"><div class="sq-panel-label">Active specialist</div>{kv_html}</div>', unsafe_allow_html=True)
    with st.expander("Method details"):
        st.markdown(
            f'<div class="sq-kv"><span class="sq-k">Tool ID</span><span class="sq-v mono">{html.escape(out.tool_name)}</span></div>',
            unsafe_allow_html=True,
        )
        if out.fallback_occurred:
            st.markdown(
                f'<div class="sq-kv"><span class="sq-k">Fallback</span><span class="sq-v">{html.escape(out.fallback_reason or "unknown")}</span></div>',
                unsafe_allow_html=True,
            )


def render_rs_context_badge(out: SpecialistOutput) -> None:
    """Honest indicator for the prompt-time BigEarthNet-19 domain-vocabulary
    context layer (src/specialists/rs_context_adapter.py). Reuses the
    existing `.sq-status-badge` styling (no new CSS, no visual redesign) -
    shows "Active" with the real retrieved-term count ONLY when
    `out.raw["rs_context_adaptation"]["applied"]` is genuinely True for
    this run. When the key is absent (classical baseline path - it never
    sets this) or `applied` is False (feature disabled, or this specific
    query had no land-cover-relevant wording to match), this renders
    nothing rather than fabricate an "Active" state."""
    info = out.raw.get("rs_context_adaptation") if isinstance(out.raw, dict) else None
    if not info or not info.get("applied"):
        return
    terms = info.get("terms") or []
    count = len(terms)
    st.markdown(
        '<span class="sq-status-badge sq-status-ready">'
        '<span class="sq-status-dot"></span>'
        f"REMOTE-SENSING ADAPTATION: ACTIVE ({count} TERM{'S' if count != 1 else ''})</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Prompt-time domain vocabulary injected from the public BigEarthNet-19 "
        f"land-cover taxonomy (not fine-tuning; source: {info.get('source', '')}) - "
        "reference terms: " + ", ".join(terms)
    )


def render_rs_example_badge(out: SpecialistOutput) -> None:
    """Honest indicator for the retrieval-augmented, dataset-grounded
    context layer (src/specialists/rs_example_adapter.py) - real
    BigEarthNet.txt text records retrieved by TF-IDF, never fine-tuning.
    Unlike `render_rs_context_badge` above, this renders even when the
    layer is NOT active, because "unavailable" is itself meaningful,
    disclosed status here (per the fallback requirement that the UI must
    say so, not stay silent) - it simply does so in a muted caption rather
    than the "Active" badge style, and only when the real-model VQA path
    actually ran (the key is absent on the classical baseline path)."""
    info = out.raw.get("rs_example_adaptation") if isinstance(out.raw, dict) else None
    if not info:
        return
    if info.get("applied"):
        n = info.get("examples_retrieved", 0)
        st.markdown(
            '<span class="sq-status-badge sq-status-ready">'
            '<span class="sq-status-dot"></span>'
            f"REMOTE-SENSING ADAPTATION: RETRIEVAL-AUGMENTED DOMAIN CONTEXT ({n} EXAMPLE{'S' if n != 1 else ''})</span>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Source: {info.get('source', 'BigEarthNet.txt')} — "
            f"{info.get('dataset_records_available', 0)} local records available, "
            f"{n} retrieved for this query. Prompt-time domain adaptation; "
            "model weights unchanged."
        )
    else:
        st.caption(
            "Remote-sensing adaptation (BigEarthNet.txt retrieval): "
            "dataset-backed adaptation unavailable"
            + (" for this query." if info.get("dataset_records_available", 0) else " — no local dataset cache found.")
        )


def render_fallback_banner(out: SpecialistOutput) -> None:
    if not out.fallback_occurred:
        return
    st.markdown(
        f'<div class="sq-danger-banner"><b>Fallback used.</b> The preferred model was unavailable during this run; '
        f"showing the classical baseline result instead. ({html.escape(out.fallback_reason or 'unknown reason')})</div>",
        unsafe_allow_html=True,
    )


def render_agent_timeline(trace: ExecutionTrace) -> None:
    """Checklist-style summary of what actually happened, derived strictly
    from real ExecutionTrace fields via routing/timeline.py - never a fake
    progress animation, never a step marked done that did not really run.

    Rendered as ONE vertical dot/connector structure built from a single
    HTML string and a single st.markdown() call, so the connector line
    between steps is visually continuous rather than broken up by
    Streamlit's own inter-element spacing (each st.markdown() call is its
    own wrapped block). There is intentionally no fourth "in progress /
    active" visual state: an ExecutionTrace only exists once a run has
    already finished (successfully or not), so the only real states are
    done, failed, and not-reached - see docs/ui_design.md Revision 4."""
    steps = timeline_mod.derive(trace)
    with st.expander("Agent process timeline"):
        parts = ['<div class="sq-tl">']
        last = len(steps) - 1
        for i, step in enumerate(steps):
            cls = _STATUS_CLASS[step.status]
            detail_text = step.detail or _STATUS_FALLBACK_TEXT.get(step.status, "")
            detail_html = f'<div class="sq-tl-detail">{html.escape(detail_text)}</div>' if detail_text else ""
            connector_html = '<div class="sq-tl-connector"></div>' if i < last else ""
            parts.append(
                '<div class="sq-tl-item">'
                f'<div class="sq-tl-marker-col"><div class="sq-tl-dot {cls}"></div>{connector_html}</div>'
                f'<div class="sq-tl-body"><div class="sq-tl-label">{html.escape(step.label)}</div>{detail_html}</div>'
                "</div>"
            )
        parts.append("</div>")
        st.markdown("".join(parts), unsafe_allow_html=True)


_AERIS_ORCHESTRATION_LABELS = {
    "Input validated": "INPUT VALIDATED",
    "Query interpreted": "QUERY UNDERSTOOD",
    "Specialist selected": "TASK SELECTED",
    "Analysis executed": "SPECIALIST EXECUTED",
    "Evidence generated": "EVIDENCE GENERATED",
    "Confidence computed": "CONFIDENCE COMPUTED",
    "Result ready": "RESULT READY",
}


def render_aeris_orchestration(trace: ExecutionTrace) -> None:
    """Phase 6 hero element: "AERIS Orchestration". This is a pure
    relabeling of the same real `routing/timeline.py:derive(trace)` steps
    `render_agent_timeline` already renders (identical status/detail
    values, identical dot/connector markup) - shown expanded and prominent
    in the unified Ask AERIS tab instead of collapsed in an expander,
    because the pivot brief calls this the most important visual element.
    The routing here is deterministic (router.decide() + fixed specialist
    dispatch), so this is described as "agentic orchestration" - automatic
    routing and execution of a specialist workflow - never as an LLM
    agent making the routing decision."""
    steps = timeline_mod.derive(trace)
    parts = [
        '<div class="aeris-orch-wrap"><div class="aeris-orch-title">AERIS Orchestration &middot; agentic routing and execution</div>',
        '<div class="sq-tl">',
    ]
    last = len(steps) - 1
    for i, step in enumerate(steps):
        cls = _STATUS_CLASS[step.status]
        label = _AERIS_ORCHESTRATION_LABELS.get(step.label, step.label.upper())
        detail_text = step.detail or _STATUS_FALLBACK_TEXT.get(step.status, "")
        detail_html = f'<div class="sq-tl-detail">{html.escape(detail_text)}</div>' if detail_text else ""
        connector_html = '<div class="sq-tl-connector"></div>' if i < last else ""
        parts.append(
            '<div class="sq-tl-item">'
            f'<div class="sq-tl-marker-col"><div class="sq-tl-dot {cls}"></div>{connector_html}</div>'
            f'<div class="sq-tl-body"><div class="sq-tl-label">{html.escape(label)}</div>{detail_html}</div>'
            "</div>"
        )
    parts.append("</div></div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_mission_brief(trace: ExecutionTrace, out_dir: str) -> None:
    """Phase 4 hero result card: "Mission Brief". Every field is read
    directly off the real `ExecutionTrace` / `SpecialistOutput` the router
    and specialist actually produced for this run - Question is the real
    query, Inputs/Sensors come from `trace.input_summary`, Analysis path is
    the real router decision, Key finding is the specialist's own real
    (truncated) answer text, Evidence is the real evidence-item count,
    Confidence is the real confidence value/label already computed
    elsewhere on this page, and Runtime is the specialist's own real
    `latency_seconds`. Nothing here is a second computation - it is a
    compact, glanceable summary of fields already on `trace`."""
    out = trace.specialist_output
    decision = trace.router_decision
    if out is None:
        return
    sensors = []
    for s in trace.input_summary:
        modality = s.get("modality") if isinstance(s, dict) else None
        if modality:
            sensors.append(str(modality).upper())
    analysis_path = TASK_TYPE_LABELS.get(decision.task_type, decision.task_type) if decision else "—"
    conf = out.confidence
    conf_str = f"{conf.value:.2f} ({CONFIDENCE_SHORT_LABEL.get(conf.method_version, conf.method_version)})" if conf else "—"
    rows = [
        ("Inputs", str(len(trace.input_summary))),
        ("Sensors", " + ".join(sensors) if sensors else "—"),
        ("Analysis path", analysis_path),
        ("Evidence", f"{len(out.evidence)} item(s)"),
        ("Confidence", conf_str),
        ("Runtime", f"{out.latency_seconds:.1f}s"),
    ]
    grid_html = "".join(
        f'<div><div class="aeris-brief-k">{html.escape(k)}</div><div class="aeris-brief-v">{html.escape(v)}</div></div>'
        for k, v in rows
    )
    st.markdown(
        f"""
        <div class="aeris-brief">
          <div class="aeris-brief-title">Mission Brief</div>
          <div class="aeris-brief-k">Question</div>
          <div class="aeris-brief-v" style="margin-bottom:var(--sp-2)">{html.escape(trace.query)}</div>
          <div class="aeris-brief-k">Key finding</div>
          <div class="aeris-brief-finding">{html.escape(_truncate(out.answer_text, 280))}</div>
          <div class="aeris-brief-grid">{grid_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _clear_tab_state(prefix: str) -> None:
    """Real, functional reset for the "Return to analysis" recovery action -
    drops this tab's current result/history so the tab shows its empty
    state again on rerun. Does not touch other tabs' state."""
    for suffix in ("_trace", "_rundir", "_inputs", "_query_used", "_extra", "_history"):
        st.session_state.pop(f"{prefix}{suffix}", None)


def render_failure(trace: ExecutionTrace, prefix: str) -> bool:
    """Renders the ERROR panel (numeric HTTP-style code, plain-English what/
    why/next-step) plus a real "Return to analysis" button. Returns whether
    that button was clicked this run - the caller is responsible for
    clearing state and rerunning, so this function stays a pure renderer."""
    raw = trace.failure or "Unknown failure."
    tb = None
    if "\nTraceback (most recent call last):" in raw:
        _, _, tb_rest = raw.partition("\nTraceback (most recent call last):")
        tb = "Traceback (most recent call last):" + tb_rest

    cls = failure_mod.classify(raw)
    st.markdown(
        f"""
        <div class="sq-panel sq-panel-danger">
          <div class="sq-error-eyebrow">ERROR {cls.code} &middot; {html.escape(cls.category)}</div>
          <div class="sq-error-title">{html.escape(cls.what)}</div>
          <div class="sq-error-why">{html.escape(cls.why)}</div>
          <div class="sq-error-label">What to do</div>
          <div class="sq-error-next">{html.escape(cls.next_step)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return_clicked = st.button("Return to analysis", key=f"{prefix}_return", type="secondary")
    if tb:
        with st.expander("Technical details"):
            st.code(tb, language="text")
    return return_clicked


def render_export_row(trace: ExecutionTrace, out_dir: str) -> None:
    """PDF is the visually stronger (primary/left) export action - it is the
    substantially-expanded, real-data-only report meant for sharing; JSON
    (secondary/right) is the raw machine-readable trace, not the default
    presentation - see docs/ui_design.md Revision 4."""
    st.markdown('<div class="sq-panel-label" style="margin-top:var(--sp-2)">Export results</div>', unsafe_allow_html=True)
    col_pdf, col_json = st.columns(2)
    with col_pdf:
        if report.reportlab_available():
            pdf_path = report.export_pdf(trace, os.path.join(out_dir, "audit_report.pdf"))
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "PDF report", f.read(), file_name="audit_report.pdf", mime="application/pdf",
                    key=f"dl_pdf_{out_dir}", type="primary",
                )
        else:
            st.caption("PDF export unavailable (reportlab not installed).")
    with col_json:
        json_path = report.export_json(trace, os.path.join(out_dir, "trace.json"))
        with open(json_path, "rb") as f:
            st.download_button(
                "JSON", f.read(), file_name="trace.json", mime="application/json",
                key=f"dl_json_{out_dir}", type="secondary",
            )


def render_technical_details(trace: ExecutionTrace) -> None:
    """Deep disclosure: the full human-readable Query->Router->Specialist->
    Evidence flow plus the raw trace JSON - collapsed by default, distinct
    from the checklist-style 'Agent process timeline' above it."""
    with st.expander("Technical details"):
        d = trace.router_decision
        st.markdown(
            f'<div class="sq-trace-flow"><span class="step-label">Query</span>{html.escape(trace.query)}</div>',
            unsafe_allow_html=True,
        )
        if d is not None:
            role = TASK_TYPE_LABELS.get(d.task_type, d.task_type)
            st.markdown(
                f'<div class="sq-trace-flow"><span class="arrow">&darr;</span></div>'
                f'<div class="sq-trace-flow"><span class="step-label">Router</span>'
                f'Routed to {html.escape(role)} (confidence {d.confidence:.2f})</div>',
                unsafe_allow_html=True,
            )
            for r in d.reasoning:
                st.markdown(f'<div class="sq-trace-reasoning">&#8226; {html.escape(r)}</div>', unsafe_allow_html=True)
        if trace.specialist_output:
            out = trace.specialist_output
            model_bit = out.model_name if out.model_name else "classical baseline"
            st.markdown(
                f'<div class="sq-trace-flow"><span class="arrow">&darr;</span></div>'
                f'<div class="sq-trace-flow"><span class="step-label">Specialist</span>'
                f'<span class="mono" style="font-family:var(--font-mono)">{html.escape(out.tool_name)}</span> &middot; {html.escape(model_bit)}</div>'
                f'<div class="sq-trace-flow"><span class="arrow">&darr;</span></div>'
                f'<div class="sq-trace-flow"><span class="step-label">Evidence</span>'
                f'{len(out.evidence)} item(s)</div>'
                f'<div class="sq-trace-flow"><span class="arrow">&darr;</span></div>'
                f'<div class="sq-trace-flow"><span class="step-label">Export</span>JSON / PDF available above</div>',
                unsafe_allow_html=True,
            )
        with st.expander("Raw data (JSON)"):
            st.json(trace.to_dict())


def render_followup(prefix: str) -> Tuple[str, bool]:
    st.markdown('<div class="sq-panel-label" style="margin-top:var(--sp-2)">Ask a follow-up</div>', unsafe_allow_html=True)
    col_q, col_btn = st.columns([3, 1])
    with col_q:
        followup_query = st.text_input(
            "Follow-up question", key=f"{prefix}_followup", label_visibility="collapsed",
            placeholder="Ask another question about the same image(s)...",
        )
    with col_btn:
        followup_clicked = st.button(
            "Ask", key=f"{prefix}_followup_run", disabled=not followup_query.strip(), type="primary",
        )
    return followup_query, followup_clicked


def render_conversation_history(prefix: str) -> None:
    """Renders prior real exchanges in this tab as a "You: / AERIS:"
    transcript - continuing one investigation on the same image/query
    context, not a generic chatbot log. Every line here is a real prior
    query and a real prior trace's own answer text (or its real failure
    message) - there is no synthetic chat history."""
    history = st.session_state.get(f"{prefix}_history", [])
    if not history:
        return
    label = f"Analysis session ({len(history)} prior exchange{'s' if len(history) != 1 else ''})"
    with st.expander(label):
        parts = []
        for q, tr in history:
            if tr.specialist_output is not None:
                answer_bit = tr.specialist_output.answer_text
            elif tr.failure:
                answer_bit = f"Analysis failed: {tr.failure.splitlines()[0]}"
            else:
                answer_bit = ""
            parts.append(
                f'<div class="sq-convo-turn"><span class="sq-convo-who">You</span>{html.escape(q)}</div>'
                f'<div class="sq-convo-turn sq-convo-answer"><span class="sq-convo-who">AERIS</span>'
                f"{html.escape(_truncate(answer_bit, 240))}</div>"
            )
        st.markdown("".join(parts), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# "Ask AERIS" (formerly "Ask SatQuery") - unified tab. Priority 1/2/4/5/7 per the agentic-analyst
# upgrade brief. This section adds NO new decision-making: it collects
# inputs in one form, calls the exact same `pipeline.run_query()` (and
# therefore the exact same `router.decide()`) every other tab already
# calls, and renders the exact same `ExecutionTrace` / `RouterDecision` /
# `SpecialistOutput` objects - just in a different, more compact,
# evidence-first layout. See `_execute_and_record`'s `mode="auto"` handling
# above for the one small, backward-compatible addition this required.
# ---------------------------------------------------------------------------

_ASK_PLAN_LABELS = {
    "Input validated": "INPUT CHECK",
    "Query interpreted": "INTENT DETECTED",
    "Specialist selected": "SPECIALIST SELECTED",
    "Analysis executed": "ANALYSIS",
    "Evidence generated": "EVIDENCE",
    "Result ready": "RESULT",
}


def _ask_preview_dir() -> str:
    d = os.path.join(RUNS_DIR, "ask_preview")
    os.makedirs(d, exist_ok=True)
    return d


def _ask_preview_metadata(path: str, declared_modality: Optional[str], declared_date: Optional[str]):
    """Loads a real file and runs it through the exact same
    `ingestion.metadata.inspect()` every specialist's input already goes
    through - used here only to populate the pre-run "Image intelligence"
    card, never to make a routing/validation decision (that only ever
    happens once, inside `pipeline.run_query()`, when Analyze is clicked).
    Returns None (never a guessed metadata object) if the file can't be
    read at all."""
    try:
        raster = raster_io.load(path)
    except raster_io.RasterLoadError:
        return None
    return metadata_mod.inspect(raster, declared_modality, declared_date)


def render_image_intelligence(meta_list: List[Any]) -> None:
    """Phase 5 "Sensor Intelligence" (formerly "Image intelligence"):
    whatever can be honestly, genuinely known about the provided image(s)
    before analysis even runs - real dimensions, band count, dtype, file
    extension, modality (declared/heuristic/unknown - the same three-state
    honesty `ingestion/metadata.py` already guarantees), CRS presence, and
    (for a pair) whether the two inputs share dimensions. Nothing here is
    guessed; a missing field renders the same "Metadata unavailable" string
    the rest of the app already uses. For a declared optical+SAR pair this
    also shows a plain OPTICAL + SAR -> MULTIMODAL ANALYSIS READY readiness
    line - it reports that both modalities are present and dimension
    pairing was checked, never that the two rasters are co-registered
    (this app never establishes or claims sub-pixel co-registration)."""
    if not meta_list:
        return
    rows: List[Tuple[str, str]] = [("Image count", str(len(meta_list)))]
    for i, m in enumerate(meta_list, start=1):
        ext = (os.path.splitext(m.path)[1].lstrip(".").upper()) or "unknown"
        rows.append((f"Image {i}", f"{m.width}x{m.height}px · {m.band_count} band(s) · {m.dtype} · {ext}"))
        rows.append((f"Image {i} modality", f"{m.modality} ({m.modality_confidence})"))
        rows.append((f"Image {i} CRS", m.crs or "Metadata unavailable"))
    same_dims = None
    if len(meta_list) == 2:
        same_dims = (meta_list[0].width, meta_list[0].height) == (meta_list[1].width, meta_list[1].height)
        rows.append(("Pairing", "same dimensions" if same_dims else "mismatched dimensions (will be resized to align)"))
    kv_html = "".join(
        f'<div class="sq-kv"><span class="sq-k">{html.escape(k)}</span><span class="sq-v">{html.escape(v)}</span></div>'
        for k, v in rows
    )
    st.markdown(f'<div class="sq-panel-label">Sensor Intelligence</div><div class="sq-panel">{kv_html}</div>', unsafe_allow_html=True)

    if len(meta_list) == 2:
        modalities = {meta_list[0].modality, meta_list[1].modality}
        if modalities == {"optical", "sar"}:
            dims_note = "dimensions match" if same_dims else "dimensions differ - will be resized to align, not co-registered"
            st.markdown(
                f'<div class="aeris-sensor-ready">OPTICAL\n  +\nSAR\n  &darr;\nMULTIMODAL ANALYSIS READY ({html.escape(dims_note)})</div>',
                unsafe_allow_html=True,
            )


def render_ask_plan(trace: ExecutionTrace) -> None:
    """Priority 1: the QUERY -> INPUT CHECK -> INTENT DETECTED -> SPECIALIST
    SELECTED -> ANALYSIS -> EVIDENCE -> RESULT flow, shown BEFORE the
    result. This is a pure relabeling of `routing/timeline.py`'s own
    `derive()` output (the identical real per-step status/detail every
    other trace-timeline view in this app already uses) - it recomputes
    nothing and cannot mark a stage done that timeline.derive() did not
    already mark done."""
    steps = timeline_mod.derive(trace)
    parts = [
        '<div class="sq-trace-flow"><span class="step-label">QUERY</span>'
        f"{html.escape(trace.query)}</div>"
    ]
    for step in steps:
        label = _ASK_PLAN_LABELS.get(step.label)
        if label is None:  # "Confidence computed" - folded into the Reliability panel below, not dropped
            continue
        color = {"done": "var(--accent2)", "failed": "var(--danger)", "not_reached": "var(--text-faint)"}[step.status]
        detail_text = step.detail or _STATUS_FALLBACK_TEXT.get(step.status, "")
        parts.append('<div class="sq-trace-flow"><span class="arrow">&darr;</span></div>')
        parts.append(
            f'<div class="sq-trace-flow"><span class="step-label" style="color:{color}">{html.escape(label)}</span>'
            f"{html.escape(detail_text)}</div>"
        )
    st.markdown('<div class="sq-panel-label">Plan</div>', unsafe_allow_html=True)
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_why_this_result(trace: ExecutionTrace) -> None:
    """Priority 2 "WHY THIS RESULT?": 1-3 short factual statements, each
    read directly off a real field (the router's own reasoning text, the
    specialist's own evidence description, or the specialist's own raw
    computed signal score) - never newly authored copy about THIS result."""
    out = trace.specialist_output
    decision = trace.router_decision
    statements: List[str] = []
    if decision is not None and decision.reasoning:
        statements.append(decision.reasoning[-1])
    if out is not None and out.evidence:
        statements.append(out.evidence[0].description)
    raw = out.raw if (out is not None and isinstance(out.raw, dict)) else {}
    if "otsu_separation_score" in raw:
        statements.append(
            f"Change/no-change separation score (Otsu): {raw['otsu_separation_score']:.3f} "
            "(0 = no separation, 1 = perfectly bimodal)."
        )
    elif "survival_ratio" in raw and raw.get("target"):
        statements.append(
            f"{raw['survival_ratio'] * 100:.1f}% of the raw colour-thresholded pixels for "
            f"target '{raw['target']}' survived cleanup."
        )
    elif "silhouette_score" in raw:
        statements.append(f"Joint optical+SAR cluster silhouette (separation quality): {raw['silhouette_score']:.3f}.")
    statements = statements[:3]
    if not statements:
        return
    st.markdown('<div class="sq-panel-label" style="margin-top:var(--sp-2)">Why this result?</div>', unsafe_allow_html=True)
    items_html = "".join(f'<div class="sq-kv"><span class="sq-v">&#8226; {html.escape(s)}</span></div>' for s in statements)
    st.markdown(items_html, unsafe_allow_html=True)


def render_what_was_used(trace: ExecutionTrace) -> None:
    """Priority 2 "WHAT WAS USED": input count, detected/declared modality
    per input, specialist name, model/tool name, and a handful of the
    specialist's own real scalar parameter values - every row read
    directly off `trace`/`SpecialistOutput` fields, nothing invented."""
    out = trace.specialist_output
    decision = trace.router_decision
    rows: List[Tuple[str, str]] = [("Inputs", str(len(trace.input_summary)))]
    for i, s in enumerate(trace.input_summary, start=1):
        modality = s.get("modality", "unknown") if isinstance(s, dict) else "unknown"
        basis = s.get("modality_basis", "unknown") if isinstance(s, dict) else "unknown"
        rows.append((f"Image {i} modality", f"{modality} ({basis})"))
    if decision is not None:
        rows.append(("Specialist", TASK_TYPE_LABELS.get(decision.task_type, decision.task_type)))
        rows.append(("Tool ID", decision.tool_name))
    if out is not None:
        rows.append(("Model", out.model_name if out.model_name else "Classical CV baseline (no pretrained model)"))
        if isinstance(out.raw, dict):
            for k, v in out.raw.items():
                if len(rows) >= 12:
                    break
                if isinstance(v, bool) or not isinstance(v, (int, float, str)):
                    continue
                label = k.replace("_", " ").capitalize()
                rows.append((label, f"{v:.3f}" if isinstance(v, float) else str(v)))
    kv_html = "".join(
        f'<div class="sq-kv"><span class="sq-k">{html.escape(k)}</span><span class="sq-v">{html.escape(v)}</span></div>'
        for k, v in rows
    )
    st.markdown(
        f'<div class="sq-panel-label" style="margin-top:var(--sp-2)">What was used</div><div class="sq-panel">{kv_html}</div>',
        unsafe_allow_html=True,
    )


def render_change_summary(trace: ExecutionTrace) -> None:
    """Priority 4: a compact bi-temporal change summary line, shown only
    when the router actually chose bitemporal_change - every number in it
    (changed-area percentage, region count, dimension-match status) is read
    straight off `SpecialistOutput.raw`/`trace.input_summary`, the same real
    fields tool_change_v0 and ingestion/metadata.py already populate."""
    out = trace.specialist_output
    raw = out.raw if isinstance(out.raw, dict) else {}
    dims = [s.get("dimensions") if isinstance(s, dict) else None for s in trace.input_summary]
    aligned = "same dimensions" if len(dims) == 2 and dims[0] == dims[1] and dims[0] is not None else \
        "mismatched input dimensions, resized to align"
    parts = []
    if raw.get("changed_fraction") is not None:
        parts.append(f"{raw['changed_fraction'] * 100:.1f}% of the scene changed")
    if raw.get("num_regions") is not None:
        parts.append(f"{raw['num_regions']} distinct region(s)")
    parts.append(aligned)
    parts.append(f"method: {out.tool_name} (grayscale differencing + Otsu threshold)")
    st.markdown(
        '<div class="sq-panel-label" style="margin-top:var(--sp-2)">Bi-temporal change summary</div>'
        f'<div class="sq-answer" style="padding-top:0">{html.escape(" · ".join(parts))}</div>',
        unsafe_allow_html=True,
    )


def render_ask_empty_state() -> None:
    st.markdown(
        """
        <div class="sq-empty">
          <div class="sq-empty-title">Ready for analysis</div>
          <div class="sq-empty-sub">Upload 1 image (VQA/grounding) or 2 images (bi-temporal change / optical+SAR fusion),
          ask a question, and click Analyze. AERIS routes it to the right specialist automatically.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ask_result(trace: ExecutionTrace, out_dir: str, prefix: str, extra_evidence=None) -> Tuple[Optional[str], bool]:
    """AERIS Mission Console result layout for the unified tab: the real
    plan is shown BEFORE the result, then the Mission Brief hero card and
    the AERIS Orchestration hero timeline (both new, both pure summaries of
    real `trace`/`SpecialistOutput` fields - see their own docstrings),
    then ANSWER / EVIDENCE / RELIABILITY / WHY THIS RESULT / WHAT WAS USED,
    the bi-temporal change summary when applicable, export, follow-up, and
    technical details - all via the exact same shared renderers the 4
    original tabs use, so there is one rendering path per concept, not
    two. The new AERIS Orchestration panel is the prominent, expanded
    hero version; the older collapsed "Agent process timeline" expander
    (reusing `render_agent_timeline` verbatim, unchanged from before the
    pivot) is kept alongside it as the familiar detail view, not removed."""
    render_ask_plan(trace)
    render_validation_notes(trace)
    if trace.failure:
        return_clicked = render_failure(trace, prefix)
        render_agent_timeline(trace)
        render_technical_details(trace)
        if return_clicked:
            _clear_tab_state(prefix)
            _trigger_rerun()
        return None, False

    out = trace.specialist_output
    decision = trace.router_decision
    render_run_meta(trace)
    render_fallback_banner(out)
    render_analysis_type(decision)

    render_mission_brief(trace, out_dir)
    render_aeris_orchestration(trace)

    st.markdown('<div class="sq-panel-label" style="margin-top:var(--sp-2)">Answer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sq-answer">{html.escape(out.answer_text)}</div>', unsafe_allow_html=True)

    if decision is not None and decision.task_type == "bitemporal_change":
        render_change_summary(trace)

    st.markdown('<div class="sq-panel-label" style="margin-top:var(--sp-2)">Evidence</div>', unsafe_allow_html=True)
    if extra_evidence:
        cols = st.columns(len(extra_evidence))
        for col, (caption, arr) in zip(cols, extra_evidence):
            with col:
                st.markdown(f'<div class="sq-evidence-caption">{html.escape(caption)}</div>', unsafe_allow_html=True)
                st.image(arr, use_container_width=True)
    for ev in out.evidence:
        if ev.image_path and os.path.exists(ev.image_path):
            st.image(ev.image_path, use_container_width=True)
        st.caption(ev.description)
    if decision is not None and decision.task_type == "grounding":
        st.caption("Classical remote-sensing baseline - not a foundation-model grounding system.")

    render_confidence_compact(out.confidence)  # RELIABILITY - same honest labels/caveats used everywhere else
    render_rs_context_badge(out)
    render_rs_example_badge(out)
    render_why_this_result(trace)
    render_what_was_used(trace)

    render_agent_timeline(trace)  # kept (collapsed detail view) alongside the new AERIS Orchestration hero above
    render_export_row(trace, out_dir)
    followup_query, followup_clicked = render_followup(prefix)
    render_conversation_history(prefix)
    render_technical_details(trace)
    return followup_query, followup_clicked


def render_empty_state(mode: str) -> None:
    """Mission-style empty state: what this capability needs, plus the four
    real capabilities shown as compact choices (from TAB_INFO, the same
    fixed labels used for the tab navigation) with the current one marked -
    not an invented feature, just a second surface for navigation copy that
    already exists."""
    needs = {
        "vqa": "an image and a question",
        "gr": "an image and a target to locate",
        "ch": "a before image and an after image",
        "fu": "an optical image and a SAR image",
    }
    choices_html = "".join(
        f'<span class="sq-empty-choice{" sq-empty-choice-current" if key == mode else ""}">'
        f'{html.escape(num)} {html.escape(title.upper())}</span>'
        for key, (num, title, _descriptor) in TAB_INFO.items()
    )
    st.markdown(
        f"""
        <div class="sq-empty">
          <div class="sq-empty-title">Ready for analysis</div>
          <div class="sq-empty-sub">Provide {html.escape(needs.get(mode, "the required imagery"))}, then run analysis.</div>
          <div class="sq-empty-choices">{choices_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_vqa_result(trace: ExecutionTrace, out_dir: str, prefix: str) -> Tuple[Optional[str], bool]:
    """VQA-specific hierarchy: the evidence image IS the source image being
    discussed, so it sits directly beside the answer instead of under a
    separate 'Evidence' heading far below (see docs/ui_design.md Revision 2,
    rule 3) - the rest of the vertical order (confidence -> active
    specialist -> agent timeline -> export -> follow-up -> technical
    details) follows the Revision 3 result hierarchy."""
    render_analysis_type(trace.router_decision)
    render_validation_notes(trace)
    if trace.failure:
        return_clicked = render_failure(trace, prefix)
        render_agent_timeline(trace)
        render_technical_details(trace)
        if return_clicked:
            _clear_tab_state(prefix)
            _trigger_rerun()
        return None, False

    out = trace.specialist_output
    render_run_meta(trace)
    render_fallback_banner(out)
    col_img, col_ans = st.columns([1, 1], gap="large")
    with col_img:
        if out.evidence and out.evidence[0].image_path and os.path.exists(out.evidence[0].image_path):
            st.image(out.evidence[0].image_path, use_container_width=True)
            st.caption("Source image analyzed")
    with col_ans:
        st.markdown(f'<div class="sq-answer">{html.escape(out.answer_text)}</div>', unsafe_allow_html=True)
        render_confidence_compact(out.confidence)
        render_active_specialist(out)
        render_rs_context_badge(out)
    render_rs_example_badge(out)

    render_agent_timeline(trace)
    render_export_row(trace, out_dir)
    followup_query, followup_clicked = render_followup(prefix)
    render_conversation_history(prefix)
    render_technical_details(trace)
    return followup_query, followup_clicked


def render_capability_result(
    trace: ExecutionTrace, out_dir: str, mode: str, prefix: str, extra_evidence=None
) -> Tuple[Optional[str], bool]:
    """Shared hierarchy for grounding / change / fusion: answer -> evidence
    (hero imagery, raw before/after or optical/SAR panels above the
    specialist's own composed evidence image) -> confidence -> active
    specialist -> agent timeline -> export -> follow-up -> technical details."""
    render_analysis_type(trace.router_decision)
    render_validation_notes(trace)
    if trace.failure:
        return_clicked = render_failure(trace, prefix)
        render_agent_timeline(trace)
        render_technical_details(trace)
        if return_clicked:
            _clear_tab_state(prefix)
            _trigger_rerun()
        return None, False

    out = trace.specialist_output
    render_run_meta(trace)
    render_fallback_banner(out)

    if mode == "change":
        changed_fraction = out.raw.get("changed_fraction") if isinstance(out.raw, dict) else None
        if changed_fraction is not None and round(changed_fraction * 100, 1) == 0.0:
            st.markdown('<div class="sq-clean-state">No significant change detected</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="sq-answer">{html.escape(out.answer_text)}</div>', unsafe_allow_html=True)

    if extra_evidence:
        cols = st.columns(len(extra_evidence))
        for col, (caption, arr) in zip(cols, extra_evidence):
            with col:
                st.markdown(f'<div class="sq-evidence-caption">{html.escape(caption)}</div>', unsafe_allow_html=True)
                st.image(arr, use_container_width=True)

    evidence_caption = {
        "grounding": "Detected regions",
        "fusion": "Fused analysis",
        "change": "Change map (before | after | highlighted)",
    }.get(mode, "Evidence")
    for ev in out.evidence:
        st.markdown(f'<div class="sq-evidence-caption">{html.escape(evidence_caption)}</div>', unsafe_allow_html=True)
        if ev.image_path and os.path.exists(ev.image_path):
            st.image(ev.image_path, use_container_width=True)
        st.caption(ev.description)

    if mode == "grounding":
        st.caption("Classical remote-sensing baseline - not a foundation-model grounding system.")

    render_confidence_compact(out.confidence)
    render_active_specialist(out)
    render_agent_timeline(trace)
    render_export_row(trace, out_dir)
    followup_query, followup_clicked = render_followup(prefix)
    render_conversation_history(prefix)
    render_technical_details(trace)
    return followup_query, followup_clicked


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

render_header()
render_sidebar()

st.markdown(
    '<div class="aeris-eyebrow">Mission Console</div>'
    '<div class="aeris-subtitle">One interface, multiple sensors and modalities - evidence-first, orchestration-observable.</div>',
    unsafe_allow_html=True,
)
render_aeris_analysis_cards()

tab_ask, tab_vqa, tab_grounding, tab_change, tab_fusion = st.tabs(
    ["🛰️ · ASK AERIS", "01 · SINGLE IMAGE", "02 · GROUNDING", "03 · BI-TEMPORAL CHANGE", "04 · OPTICAL + SAR"]
)

# Demo-mode presets (Phase 11) - map directly onto the pre-existing
# sample_path1/sample_path2/sample_modality1/sample_modality2/sample_date1/
# sample_date2 variables `_execute_and_record` already consumes, so this is
# the exact same tested code path as the older "Sample ..." entries, just
# pointed at the real local Sentinel-1/Sentinel-2 derived PNGs for the
# optical+SAR and VQA presets. Nothing here downloads or fabricates data -
# see demo_assets/REAL_OPTICAL_SAR/source.txt for real provenance.
REAL_OPTICAL_SAR_DIR = os.path.join(REPO_ROOT, "demo_assets", "REAL_OPTICAL_SAR")
_REAL_ASSETS_PRESENT = os.path.isfile(os.path.join(REAL_OPTICAL_SAR_DIR, "optical_real.png")) and os.path.isfile(
    os.path.join(REAL_OPTICAL_SAR_DIR, "sar_real.png")
)

# --- VIEW 0: Ask AERIS (unified Mission Console) -------------------------------
# Priority 1 (must-have): one form, one query, one Analyze button, routed
# automatically by the real router.decide() every other tab already uses.
# The four tabs below remain the reliable, capability-specific manual
# fallback - unchanged, and positioned after this one.
with tab_ask:
    prefix = "ask"
    col_in, col_out = st.columns([1, 2], gap="large")
    with col_in:
        st.markdown(
            '<div class="sq-nav-eyebrow">00 &middot; INPUT</div>'
            '<div class="sq-nav-descriptor">Ask AERIS — describe what you need from the imagery</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Upload 1 image for single-image analysis (VQA/grounding), or 2 images for "
            "bi-temporal change / optical+SAR fusion - AERIS routes to the right "
            "specialist automatically. The 4 tabs to the right are the reliable, "
            "manual, capability-specific fallback for each analysis type."
        )

        sample_options = ["None - upload my own", "Single sample image", "Sample before/after pair", "Sample optical+SAR pair"]
        if _REAL_ASSETS_PRESENT:
            sample_options.append("AERIS Demo: Optical + SAR Mission (real Sentinel-1/Sentinel-2)")
            sample_options.append("AERIS Demo: VQA Mission (real Sentinel-2 optical)")
        sample_choice = st.selectbox(
            "Use bundled sample data (optional)",
            sample_options,
            key="ask_sample",
        )
        using_sample = sample_choice != "None - upload my own"

        sample_path1 = sample_path2 = None
        sample_modality1 = sample_modality2 = None
        sample_date1 = sample_date2 = None
        if sample_choice == "Single sample image":
            sample_path1 = os.path.join(FIXTURES_DIR, "single_image.png")
        elif sample_choice == "Sample before/after pair":
            sample_path1 = os.path.join(FIXTURES_DIR, "change_before.png")
            sample_path2 = os.path.join(FIXTURES_DIR, "change_after.png")
            sample_date1 = (datetime.date.today() - datetime.timedelta(days=365)).isoformat()
            sample_date2 = datetime.date.today().isoformat()
        elif sample_choice == "Sample optical+SAR pair":
            sample_path1 = os.path.join(FIXTURES_DIR, "fusion_optical.png")
            sample_path2 = os.path.join(FIXTURES_DIR, "fusion_sar.png")
            sample_modality1, sample_modality2 = "optical", "sar"
        elif sample_choice.startswith("AERIS Demo: Optical + SAR Mission"):
            sample_path1 = os.path.join(REAL_OPTICAL_SAR_DIR, "optical_real.png")
            sample_path2 = os.path.join(REAL_OPTICAL_SAR_DIR, "sar_real.png")
            sample_modality1, sample_modality2 = "optical", "sar"
        elif sample_choice.startswith("AERIS Demo: VQA Mission"):
            sample_path1 = os.path.join(REAL_OPTICAL_SAR_DIR, "optical_real.png")
            sample_modality1 = "optical"

        if using_sample and sample_choice.startswith("AERIS Demo:"):
            st.markdown(
                '<div class="sq-sample-note">Real Sentinel-1/Sentinel-2 derived imagery (see demo_assets/REAL_OPTICAL_SAR/source.txt) - not synthetic.</div>',
                unsafe_allow_html=True,
            )
            ask_upload1 = ask_upload2 = None
        elif using_sample:
            st.markdown('<div class="sq-sample-note">Synthetic sample data - not real satellite imagery.</div>', unsafe_allow_html=True)
            ask_upload1 = ask_upload2 = None
        else:
            ask_upload1 = st.file_uploader("Image 1", type=["png", "jpg", "jpeg", "tif", "tiff"], key="ask_upload1")
            ask_upload2 = st.file_uploader(
                "Image 2 (optional - for change/fusion)", type=["png", "jpg", "jpeg", "tif", "tiff"], key="ask_upload2"
            )

        ask_modality1_sel = st.selectbox("Image 1 modality (optional)", ["Not declared", "optical", "sar"], key="ask_modality1")
        ask_modality2_sel = st.selectbox("Image 2 modality (optional)", ["Not declared", "optical", "sar"], key="ask_modality2")
        ask_declare_dates = st.checkbox("Declare acquisition dates (helps disambiguate bi-temporal change)", key="ask_declare_dates")
        if ask_declare_dates:
            ask_date1_widget = st.date_input(
                "Image 1 date", value=datetime.date.today() - datetime.timedelta(days=365), key="ask_date1"
            )
            ask_date2_widget = st.date_input("Image 2 date", value=datetime.date.today(), key="ask_date2")
        else:
            ask_date1_widget = ask_date2_widget = None

        # Priority 5: Image intelligence, computed from whatever is currently
        # provided - BEFORE Analyze is clicked - via the exact same
        # ingestion.metadata.inspect() every real run already uses.
        preview_meta = []
        if using_sample:
            if sample_path1:
                m1 = _ask_preview_metadata(sample_path1, sample_modality1, sample_date1)
                if m1:
                    preview_meta.append(m1)
            if sample_path2:
                m2 = _ask_preview_metadata(sample_path2, sample_modality2, sample_date2)
                if m2:
                    preview_meta.append(m2)
        else:
            if ask_upload1 is not None:
                p1 = _save_upload(ask_upload1, _ask_preview_dir(), "preview1")
                m1 = _ask_preview_metadata(
                    p1, None if ask_modality1_sel == "Not declared" else ask_modality1_sel,
                    ask_date1_widget.isoformat() if ask_date1_widget else None,
                )
                if m1:
                    preview_meta.append(m1)
            if ask_upload2 is not None:
                p2 = _save_upload(ask_upload2, _ask_preview_dir(), "preview2")
                m2 = _ask_preview_metadata(
                    p2, None if ask_modality2_sel == "Not declared" else ask_modality2_sel,
                    ask_date2_widget.isoformat() if ask_date2_widget else None,
                )
                if m2:
                    preview_meta.append(m2)
        render_image_intelligence(preview_meta)

        render_mission_selector("ask_query")
        render_example_chips("ask", "ask_query")
        st.markdown('<div class="sq-ask-label">Ask your question</div>', unsafe_allow_html=True)
        query = _seeded_text_input(
            "Query", "ask_query", EXAMPLE_QUERIES["ask"][0], label_visibility="collapsed",
        )

        have_image1 = (using_sample and sample_path1 is not None) or (not using_sample and ask_upload1 is not None)
        note = None
        if not have_image1:
            note = ("INSUFFICIENT INPUTS", "Add at least one image, or choose a bundled sample.")
        elif not query.strip():
            note = ("INSUFFICIENT INPUTS", "Enter a query.")
        render_validation_note(note)
        run_clicked = st.button("Analyze", key="ask_analyze_btn", disabled=note is not None, type="primary")

    if run_clicked:
        run_dir = _new_run_dir("ask")
        if using_sample:
            img1_path, img2_path = sample_path1, sample_path2
            mod1, mod2 = sample_modality1, sample_modality2
            d1, d2 = sample_date1, sample_date2
        else:
            img1_path = _save_upload(ask_upload1, run_dir, "input1")
            img2_path = _save_upload(ask_upload2, run_dir, "input2") if ask_upload2 is not None else None
            mod1 = None if ask_modality1_sel == "Not declared" else ask_modality1_sel
            mod2 = None if ask_modality2_sel == "Not declared" else ask_modality2_sel
            d1 = ask_date1_widget.isoformat() if ask_date1_widget else None
            d2 = ask_date2_widget.isoformat() if ask_date2_widget else None
        inputs: Dict[str, Any] = {"image1_path": img1_path}
        if img2_path:
            inputs["image2_path"] = img2_path
        if mod1:
            inputs["modality1"] = mod1
        if mod2:
            inputs["modality2"] = mod2
        if d1:
            inputs["date1"] = d1
        if d2:
            inputs["date2"] = d2
        _execute_and_record(prefix, "auto", inputs, query, run_dir=run_dir)

    followup_query, followup_clicked = None, False
    with col_out:
        st.markdown('<div class="sq-nav-eyebrow">RESULT</div>', unsafe_allow_html=True)
        if f"{prefix}_trace" in st.session_state:
            followup_query, followup_clicked = render_ask_result(
                st.session_state[f"{prefix}_trace"], st.session_state[f"{prefix}_rundir"], prefix,
                extra_evidence=st.session_state.get(f"{prefix}_extra"),
            )
        else:
            render_ask_empty_state()

    if followup_clicked and followup_query and followup_query.strip():
        _execute_and_record(prefix, "auto", st.session_state.get(f"{prefix}_inputs", {}), followup_query)
        _trigger_rerun()

# --- VIEW 1: Single-image VQA -------------------------------------------------
with tab_vqa:
    prefix = "vqa"
    col_in, col_out = st.columns([1, 2], gap="large")
    with col_in:
        render_section_header(prefix, "input")
        render_mode_help(prefix)
        use_fixture = st.checkbox("Use a bundled sample image", key="vqa_fixture")
        if use_fixture:
            st.markdown('<div class="sq-sample-note">Synthetic sample - not real satellite imagery.</div>', unsafe_allow_html=True)
            upload = None
        else:
            upload = st.file_uploader("Image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="vqa_upload")
        modality = st.selectbox("Modality (optional)", ["Not declared", "optical", "sar"], key="vqa_modality")
        render_example_chips("vqa", "vqa_query")
        st.markdown('<div class="sq-ask-label">Ask your question</div>', unsafe_allow_html=True)
        query = _seeded_text_input(
            "Query", "vqa_query", "Describe the major land cover types visible in this image.",
            label_visibility="collapsed",
        )

        note = None
        if not (use_fixture or upload):
            note = ("INSUFFICIENT INPUTS", "Add an image or use the bundled sample.")
        elif not query.strip():
            note = ("INSUFFICIENT INPUTS", "Enter a query.")
        render_validation_note(note)
        run_clicked = st.button("Run analysis", key="vqa_run", disabled=note is not None, type="primary")
        if not _capability_ready([SMOLVLM_TOOL_NAME]):
            st.caption("The vision-language model isn't available on this machine - the classical baseline will answer instead.")

    if run_clicked:
        run_dir = _new_run_dir("vqa")
        image_path = os.path.join(FIXTURES_DIR, "single_image.png") if use_fixture else _save_upload(upload, run_dir, "input1")
        inputs = {"image1_path": image_path, "modality1": None if modality == "Not declared" else modality}
        _execute_and_record(prefix, "vqa", inputs, query, run_dir=run_dir)

    followup_query, followup_clicked = None, False
    with col_out:
        render_section_header(prefix, "result")
        if f"{prefix}_trace" in st.session_state:
            followup_query, followup_clicked = render_vqa_result(
                st.session_state[f"{prefix}_trace"], st.session_state[f"{prefix}_rundir"], prefix
            )
        else:
            render_empty_state("vqa")

    if followup_clicked and followup_query and followup_query.strip():
        _execute_and_record(prefix, "vqa", st.session_state.get(f"{prefix}_inputs", {}), followup_query)
        _trigger_rerun()

# --- VIEW 2: Grounding ---------------------------------------------------------
with tab_grounding:
    prefix = "gr"
    col_in, col_out = st.columns([1, 2], gap="large")
    with col_in:
        render_section_header(prefix, "input")
        render_mode_help(prefix)
        use_fixture = st.checkbox("Use a bundled sample image", key="gr_fixture")
        if use_fixture:
            st.markdown('<div class="sq-sample-note">Synthetic sample - not real satellite imagery.</div>', unsafe_allow_html=True)
            upload = None
        else:
            upload = st.file_uploader("Image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="gr_upload")
        modality = st.selectbox("Modality (optional)", ["Not declared", "optical", "sar"], key="gr_modality")
        st.caption("Recognized targets: water, vegetation, urban / built-up.")
        render_example_chips("gr", "gr_query")
        st.markdown('<div class="sq-ask-label">Ask your question</div>', unsafe_allow_html=True)
        query = _seeded_text_input(
            "Query", "gr_query", "Locate the water body in this image and highlight it.",
            label_visibility="collapsed",
        )

        note = None
        if not (use_fixture or upload):
            note = ("INSUFFICIENT INPUTS", "Add an image or use the bundled sample.")
        elif not query.strip():
            note = ("INSUFFICIENT INPUTS", "Enter a query.")
        render_validation_note(note)
        run_clicked = st.button("Run analysis", key="gr_run", disabled=note is not None, type="primary")

    if run_clicked:
        run_dir = _new_run_dir("grounding")
        image_path = os.path.join(FIXTURES_DIR, "single_image.png") if use_fixture else _save_upload(upload, run_dir, "input1")
        inputs = {"image1_path": image_path, "modality1": None if modality == "Not declared" else modality}
        _execute_and_record(prefix, "grounding", inputs, query, run_dir=run_dir)

    followup_query, followup_clicked = None, False
    with col_out:
        render_section_header(prefix, "result")
        if f"{prefix}_trace" in st.session_state:
            followup_query, followup_clicked = render_capability_result(
                st.session_state[f"{prefix}_trace"], st.session_state[f"{prefix}_rundir"], mode="grounding", prefix=prefix
            )
        else:
            render_empty_state("gr")

    if followup_clicked and followup_query and followup_query.strip():
        _execute_and_record(prefix, "grounding", st.session_state.get(f"{prefix}_inputs", {}), followup_query)
        _trigger_rerun()

# --- VIEW 3: Bi-temporal change -----------------------------------------------
with tab_change:
    prefix = "ch"
    col_in, col_out = st.columns([1, 2], gap="large")
    with col_in:
        render_section_header(prefix, "input")
        render_mode_help(prefix)
        use_fixture = st.checkbox("Use a bundled sample pair", key="ch_fixture")
        if use_fixture:
            st.markdown('<div class="sq-sample-note">Synthetic before/after pair - not real satellite imagery.</div>', unsafe_allow_html=True)
            upload1 = upload2 = None
        else:
            upload1 = st.file_uploader("Before image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="ch_upload1")
            upload2 = st.file_uploader("After image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="ch_upload2")
        default_date1 = datetime.date.today() - datetime.timedelta(days=365)
        date1 = st.date_input("Before - date", value=default_date1, key="ch_date1")
        date2 = st.date_input("After - date", value=datetime.date.today(), key="ch_date2")
        render_example_chips("ch", "ch_query")
        st.markdown('<div class="sq-ask-label">Ask your question</div>', unsafe_allow_html=True)
        query = _seeded_text_input(
            "Query", "ch_query", "What changed between these two dates?",
            label_visibility="collapsed",
        )

        note = None
        if use_fixture:
            pass
        elif upload1 is None and upload2 is None:
            note = ("INSUFFICIENT INPUTS", "Add a before and an after image, or use the bundled sample.")
        elif (upload1 is None) != (upload2 is None):
            note = ("MISSING SECOND IMAGE", "Add both a before and an after image to continue.")
        if note is None and not query.strip():
            note = ("INSUFFICIENT INPUTS", "Enter a query.")
        render_validation_note(note)
        have_both = use_fixture or (upload1 is not None and upload2 is not None)
        run_clicked = st.button("Run analysis", key="ch_run", disabled=not have_both or not query.strip(), type="primary")

    if run_clicked:
        run_dir = _new_run_dir("change")
        if use_fixture:
            path1 = os.path.join(FIXTURES_DIR, "change_before.png")
            path2 = os.path.join(FIXTURES_DIR, "change_after.png")
        else:
            path1 = _save_upload(upload1, run_dir, "before")
            path2 = _save_upload(upload2, run_dir, "after")
        inputs = {
            "image1_path": path1,
            "image2_path": path2,
            "date1": date1.isoformat(),
            "date2": date2.isoformat(),
        }
        _execute_and_record(prefix, "change", inputs, query, run_dir=run_dir)

    followup_query, followup_clicked = None, False
    with col_out:
        render_section_header(prefix, "result")
        if f"{prefix}_trace" in st.session_state:
            followup_query, followup_clicked = render_capability_result(
                st.session_state[f"{prefix}_trace"], st.session_state[f"{prefix}_rundir"], mode="change", prefix=prefix,
                extra_evidence=st.session_state.get(f"{prefix}_extra"),
            )
        else:
            render_empty_state("ch")

    if followup_clicked and followup_query and followup_query.strip():
        _execute_and_record(prefix, "change", st.session_state.get(f"{prefix}_inputs", {}), followup_query)
        _trigger_rerun()

# --- VIEW 4: Optical + SAR fusion ---------------------------------------------
with tab_fusion:
    prefix = "fu"
    col_in, col_out = st.columns([1, 2], gap="large")
    with col_in:
        render_section_header(prefix, "input")
        render_mode_help(prefix)
        use_fixture = st.checkbox("Use a bundled sample pair", key="fu_fixture")
        if use_fixture:
            st.markdown('<div class="sq-sample-note">Synthetic optical+SAR-like pair - not real satellite imagery.</div>', unsafe_allow_html=True)
            upload_opt = upload_sar = None
        else:
            upload_opt = st.file_uploader("Optical image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="fu_upload_opt")
            upload_sar = st.file_uploader("SAR image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="fu_upload_sar")
        render_example_chips("fu", "fu_query")
        st.markdown('<div class="sq-ask-label">Ask your question</div>', unsafe_allow_html=True)
        query = _seeded_text_input(
            "Query", "fu_query", "Use both images to identify built-up and water-covered regions.",
            label_visibility="collapsed",
        )

        note = None
        if use_fixture:
            pass
        elif upload_opt is None and upload_sar is None:
            note = ("INSUFFICIENT INPUTS", "Add an optical and a SAR image, or use the bundled sample.")
        elif (upload_opt is None) != (upload_sar is None):
            note = ("MISSING SECOND IMAGE", "Add both an optical and a SAR image to continue.")
        if note is None and not query.strip():
            note = ("INSUFFICIENT INPUTS", "Enter a query.")
        render_validation_note(note)
        have_both = use_fixture or (upload_opt is not None and upload_sar is not None)
        run_clicked = st.button("Run analysis", key="fu_run", disabled=not have_both or not query.strip(), type="primary")

    if run_clicked:
        run_dir = _new_run_dir("fusion")
        if use_fixture:
            path_opt = os.path.join(FIXTURES_DIR, "fusion_optical.png")
            path_sar = os.path.join(FIXTURES_DIR, "fusion_sar.png")
        else:
            path_opt = _save_upload(upload_opt, run_dir, "optical")
            path_sar = _save_upload(upload_sar, run_dir, "sar")
        inputs = {
            "image1_path": path_opt,
            "image2_path": path_sar,
            "modality1": "optical",
            "modality2": "sar",
        }
        _execute_and_record(prefix, "fusion", inputs, query, run_dir=run_dir)

    followup_query, followup_clicked = None, False
    with col_out:
        render_section_header(prefix, "result")
        if f"{prefix}_trace" in st.session_state:
            followup_query, followup_clicked = render_capability_result(
                st.session_state[f"{prefix}_trace"], st.session_state[f"{prefix}_rundir"], mode="fusion", prefix=prefix,
                extra_evidence=st.session_state.get(f"{prefix}_extra"),
            )
        else:
            render_empty_state("fu")

    if followup_clicked and followup_query and followup_query.strip():
        _execute_and_record(prefix, "fusion", st.session_state.get(f"{prefix}_inputs", {}), followup_query)
        _trigger_rerun()
