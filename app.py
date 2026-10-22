"""Chest X-ray pneumonia screening demo (research use only).

Streamlit UI. Model code lives in src/inference.py, decision logic in src/logic.py.
"""
from __future__ import annotations

import base64
import hashlib
import io
from itertools import zip_longest
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.inference import DISPLAY_NAMES, MODEL_NAMES, Engine, load_config
from src.logic import agreement, decide, image_checks, load_image

ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "samples"
IMG_EXT = {".jpg", ".jpeg", ".png"}

st.set_page_config(
    page_title="Chest X-ray Pneumonia Screening",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------
# Styling (semi-transparent surfaces so it works in both light and dark themes)
# --------------------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stMarkdown, .hero, .notice, .card, .verdict, .thumb, .pill, .empty, table.perf, .footer, .meter-labels
  { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }
footer { visibility: hidden; }

.hero { background: linear-gradient(135deg,#0ea5e9 0%,#6366f1 55%,#8b5cf6 100%); border-radius: 20px;
        padding: 30px 34px; color: #fff; margin-bottom: 16px; box-shadow: 0 10px 30px rgba(99,102,241,.25); }
.hero h1 { margin: 0 0 6px 0; padding: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -.02em; color: #fff; }
.hero p { margin: 0; font-size: 1.02rem; opacity: .93; max-width: 780px; color: #fff; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.chip { background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.3); padding: 4px 12px;
        border-radius: 999px; font-size: .8rem; font-weight: 600; color: #fff; }

.notice { border-left: 4px solid #f59e0b; background: rgba(245,158,11,.12); padding: 12px 16px;
          border-radius: 10px; font-size: .9rem; margin-bottom: 22px; }

.card { border: 1px solid rgba(128,128,128,.28); background: rgba(128,128,128,.06); border-radius: 16px;
        padding: 18px 20px; margin-bottom: 14px; }
.card h4 { margin: 0 0 12px 0; padding: 0; font-size: .75rem; text-transform: uppercase;
           letter-spacing: .09em; opacity: .65; font-weight: 700; }
.xray { width: 100%; max-height: 420px; object-fit: contain; border-radius: 12px; background: #000; display: block; }

.verdict { border-radius: 16px; padding: 20px 22px; border: 1px solid; margin-bottom: 14px; }
.v-pneumonia { background: rgba(239,68,68,.12); border-color: rgba(239,68,68,.55); }
.v-normal { background: rgba(34,197,94,.12); border-color: rgba(34,197,94,.55); }
.v-borderline { background: rgba(245,158,11,.14); border-color: rgba(245,158,11,.6); }
.v-top { display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; flex-wrap: wrap; }
.v-label { font-size: .75rem; text-transform: uppercase; letter-spacing: .09em; font-weight: 700; opacity: .7; }
.v-title { font-size: 1.75rem; font-weight: 800; letter-spacing: -.01em; margin: 2px 0 4px 0;
           display: flex; align-items: center; gap: 10px; }
.dot { width: 14px; height: 14px; border-radius: 50%; display: inline-block; flex: none; }
.dot-pneumonia { background: #ef4444; } .dot-normal { background: #22c55e; } .dot-borderline { background: #f59e0b; }
.v-prob { font-size: 2.6rem; font-weight: 800; line-height: 1; text-align: left; margin-top: 6px; }
.v-prob small { display: block; font-size: .72rem; font-weight: 600; opacity: .65; letter-spacing: .04em; margin-top: 8px; }
.v-sub { font-size: .9rem; opacity: .82; max-width: 520px; }

.meter { position: relative; height: 14px; margin: 22px 0 6px 0; }
.meter-track { position: absolute; inset: 0; border-radius: 999px; opacity: .85;
               background: linear-gradient(90deg,#22c55e 0%,#eab308 50%,#ef4444 100%); }
.meter .thr { position: absolute; top: -6px; bottom: -6px; width: 2px; background: currentColor; opacity: .85; }
.meter .marker { position: absolute; top: 50%; width: 20px; height: 20px; border-radius: 50%; background: #fff;
                 border: 4px solid #1f2937; transform: translate(-50%,-50%); box-shadow: 0 2px 8px rgba(0,0,0,.35); }
.meter-labels { display: flex; justify-content: space-between; font-size: .75rem; opacity: .7; }

.mrow { display: grid; grid-template-columns: 128px 1fr 62px; align-items: center; gap: 12px; margin: 11px 0; }
.mname { font-size: .9rem; font-weight: 600; }
.mbar { position: relative; height: 10px; border-radius: 999px; background: rgba(128,128,128,.25); }
.mfill { height: 100%; border-radius: 999px; }
.mthr { position: absolute; top: -4px; bottom: -4px; width: 2px; background: currentColor; opacity: .55; }
.mval { text-align: right; font-weight: 700; font-size: .9rem; font-variant-numeric: tabular-nums; }
.note { font-size: .85rem; opacity: .82; margin-top: 8px; }
.pill { display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: .8rem; font-weight: 600;
        border: 1px solid rgba(128,128,128,.4); background: rgba(128,128,128,.08); }
.empty { text-align: center; padding: 56px 20px; opacity: .8; }
.empty .big { font-size: 2.8rem; }
.empty .t { font-weight: 700; font-size: 1.1rem; margin: 6px 0 4px 0; }

.thumb img { width: 100%; aspect-ratio: 1/1; object-fit: cover; border-radius: 12px; background: #000;
             background-clip: padding-box; display: block; border: 3px solid transparent; }
.thumb.sel img { border-color: #6366f1; }
.thumb .cap { font-size: .75rem; font-weight: 600; margin: 5px 2px 6px 2px; opacity: .85; }
.stButton > button { width: 100%; border-radius: 10px; font-weight: 600; }

table.perf { width: 100%; border-collapse: collapse; font-size: .88rem; }
table.perf th { text-align: left; font-size: .72rem; text-transform: uppercase; letter-spacing: .07em; opacity: .65;
                padding: 8px 10px; border-bottom: 1px solid rgba(128,128,128,.35); }
table.perf td { padding: 9px 10px; border-bottom: 1px solid rgba(128,128,128,.18); font-variant-numeric: tabular-nums; }
.footer { text-align: center; font-size: .8rem; opacity: .6; margin: 28px 0 8px 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def h(s: str) -> str:
    """Collapse an HTML snippet to one line (indented lines would be parsed as Markdown code)."""
    return " ".join(line.strip() for line in s.strip().splitlines())


# --------------------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models…")
def get_engine() -> Engine:
    return Engine(load_config())


@st.cache_data(show_spinner=False, max_entries=32, ttl=3600)
def run_inference(data: bytes) -> dict[str, float]:
    return get_engine().predict(load_image(data))


def to_data_uri(img: Image.Image, max_side: int, quality: int = 88) -> str:
    im = img.convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@st.cache_data(show_spinner=False)
def sample_thumb(path: str, mtime: float) -> str:  # mtime busts the cache when a file changes
    return to_data_uri(load_image(Path(path).read_bytes()), 360)


def list_samples() -> list[Path]:
    groups = []
    for cls in ("normal", "pneumonia"):
        folder = SAMPLES_DIR / cls
        groups.append(sorted(p for p in folder.glob("*") if p.suffix.lower() in IMG_EXT) if folder.exists() else [])
    return [p for pair in zip_longest(*groups) for p in pair if p is not None]  # interleave so labels aren't obvious


# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------
st.markdown(h("""
<div class="hero">
  <h1>Chest X-ray Pneumonia Screening</h1>
  <p>Upload a frontal chest X-ray and three deep-learning models (ResNet-50, DenseNet-121, EfficientNet-B0)
  estimate the probability that it shows pneumonia.</p>
  <div class="chips">
    <span class="chip">3-model ensemble</span><span class="chip">Calibrated probabilities</span>
    <span class="chip">Adjustable threshold</span><span class="chip">Research demo</span>
  </div>
</div>
<div class="notice"><b>Not a medical device.</b> This is a student research project trained on one public paediatric
dataset. It must not be used to diagnose, treat or make decisions about any patient. Please do not upload real patient data.</div>
"""), unsafe_allow_html=True)

engine = get_engine()
available = engine.available()
if not available:
    st.error(
        "No model files were found in the `models/` folder. Add `resnet50.pt`, `densenet121.pt`, "
        "`efficientnet_b0.pt` and `app_config.json` (see the README), then reboot the app."
    )
    st.stop()
cfg = engine.cfg

# --------------------------------------------------------------------------------------
# Sidebar settings
# --------------------------------------------------------------------------------------
ENSEMBLE_LABEL = "Ensemble (average of all models)"
options = ([ENSEMBLE_LABEL] if len(available) > 1 else []) + [DISPLAY_NAMES[n] for n in available]
with st.sidebar:
    st.markdown("## Settings")
    choice = st.radio(
        "Model used for the result", options, index=0,
        help="The ensemble averages the calibrated probabilities of all models. "
             "All individual probabilities are always shown next to the result.",
    )
    threshold = st.slider(
        "Decision threshold", 0.05, 0.95, 0.50, 0.01,
        help="A probability at or above this value is reported as 'pneumonia likely'. "
             "Lower it to catch more pneumonia (more false alarms); raise it to reduce false alarms (more misses).",
    )
    st.caption("Results within ±0.10 of the threshold are reported as borderline.")
    st.markdown("---")
    st.caption("Images are processed in memory and are not saved by this app.")

# --------------------------------------------------------------------------------------
# Input selection
# --------------------------------------------------------------------------------------
def _use_upload() -> None:
    st.session_state["source"] = "upload"


def _pick_sample(path: str) -> None:
    st.session_state["source"] = "sample"
    st.session_state["sample_path"] = path


def current_input():
    """Return (bytes, display_name, dataset_label_or_None) for the selected image, or None."""
    source = st.session_state.get("source")
    up = st.session_state.get("uploader")
    if source == "upload" and up is not None:
        return up.getvalue(), up.name, None
    sp = st.session_state.get("sample_path")
    if source == "sample" and sp and Path(sp).exists():
        p = Path(sp)
        return p.read_bytes(), p.name, p.parent.name
    return None


left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("### 1 · Choose an X-ray")
    tab_up, tab_samples = st.tabs(["Upload", "Try a sample"])
    with tab_up:
        st.file_uploader(
            "Chest X-ray (PNG or JPG)", type=["png", "jpg", "jpeg"], key="uploader",
            on_change=_use_upload, label_visibility="collapsed",
        )
        st.caption("Frontal chest X-ray as PNG or JPG, up to 10 MB.")
    with tab_samples:
        samples = list_samples()
        if not samples:
            st.info("No sample images yet. Add X-rays to `samples/normal/` and `samples/pneumonia/` in the repository.")
        else:
            st.caption("Sample images from the Kaggle test split. Their dataset labels are revealed once you run one.")
            selected = st.session_state.get("sample_path") if st.session_state.get("source") == "sample" else None
            for row_start in range(0, len(samples), 4):
                cols = st.columns(4, gap="small")
                for col, idx in zip(cols, range(row_start, min(row_start + 4, len(samples)))):
                    p = samples[idx]
                    with col:
                        sel = " sel" if selected == str(p) else ""
                        uri = sample_thumb(str(p), p.stat().st_mtime)
                        st.markdown(
                            h(f'<div class="thumb{sel}"><img src="{uri}" alt="sample X-ray"/>'
                              f'<div class="cap">Sample {idx + 1}</div></div>'),
                            unsafe_allow_html=True,
                        )
                        st.button("Analyse", key=f"pick_{idx}", on_click=_pick_sample, args=(str(p),))

    selection = current_input()
    if selection is not None:
        data, name, truth = selection
        try:
            img = load_image(data)
        except Exception:
            st.error("That file could not be read as an image. Please try a PNG or JPG.")
            selection = None
        else:
            st.markdown(
                h(f'<div class="card"><h4>Selected image</h4><img class="xray" src="{to_data_uri(img, 900)}" '
                  f'alt="selected X-ray"/><div class="note">{name}</div></div>'),
                unsafe_allow_html=True,
            )

# --------------------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------------------
def meter_html(p: float, thr: float) -> str:
    pp, tt = min(max(p, 0.01), 0.99) * 100, min(max(thr, 0.0), 1.0) * 100
    return h(f"""
    <div class="meter"><div class="meter-track"></div>
      <div class="thr" style="left:{tt:.1f}%"></div><div class="marker" style="left:{pp:.1f}%"></div></div>
    <div class="meter-labels"><span>0%</span><span>threshold {thr * 100:.0f}%</span><span>100%</span></div>
    """)


def model_row(label: str, p: float, thr: float, bold: bool = False) -> str:
    color = "#ef4444" if p >= thr else "#22c55e"
    weight = "800" if bold else "600"
    return h(f"""
    <div class="mrow"><div class="mname" style="font-weight:{weight}">{label}</div>
      <div class="mbar"><div class="mfill" style="width:{p * 100:.1f}%;background:{color}"></div>
        <div class="mthr" style="left:{thr * 100:.1f}%"></div></div>
      <div class="mval">{p * 100:.1f}%</div></div>
    """)


with right:
    st.markdown("### 2 · Result")
    if selection is None:
        st.markdown(h("""
        <div class="card empty"><div class="big">🫁</div><div class="t">No image selected yet</div>
        <div>Upload an X-ray or pick a sample on the left to see the screening result.</div></div>
        """), unsafe_allow_html=True)
    else:
        warnings, blocking = image_checks(img)
        proceed = True
        if warnings:
            for w in warnings:
                st.warning(w)
            if blocking:
                key = "override_" + hashlib.md5(data).hexdigest()[:8]
                proceed = st.checkbox("Analyse anyway (the result will not be meaningful)", key=key)
        if proceed:
            with st.spinner("Analysing X-ray…"):
                probs = run_inference(data)
            names = [n for n in MODEL_NAMES if n in probs]
            ensemble = float(np.mean([probs[n] for n in names]))
            if choice == ENSEMBLE_LABEL:
                p, used = ensemble, "Ensemble"
            else:
                key_name = next(n for n in names if DISPLAY_NAMES[n] == choice)
                p, used = probs[key_name], choice
            verdict = decide(p, threshold)
            n_pos, n_total = agreement(probs, threshold)

            st.markdown(h(f"""
            <div class="verdict v-{verdict.key}">
              <div class="v-top">
                <div><div class="v-label">Screening result · {used}</div>
                  <div class="v-title"><span class="dot dot-{verdict.key}"></span>{verdict.title}</div>
                  <div class="v-sub">{verdict.detail}</div></div>
                <div class="v-prob">{p * 100:.1f}%<small>P(pneumonia)</small></div>
              </div>
              {meter_html(p, threshold)}
            </div>
            """), unsafe_allow_html=True)

            rows = "".join(model_row(DISPLAY_NAMES[n], probs[n], threshold) for n in names)
            if len(names) > 1:
                rows += model_row("Ensemble", ensemble, threshold, bold=True)
                if n_pos in (0, n_total):
                    agree = f"All {n_total} models agree."
                else:
                    agree = f"The models disagree: {n_pos} of {n_total} flag pneumonia at this threshold."
            else:
                agree = ""
            st.markdown(h(f"""
            <div class="card"><h4>Individual model probabilities</h4>{rows}
              <div class="note">{agree} Probabilities are temperature-calibrated on a validation set; they are estimates,
              not diagnoses.</div></div>
            """), unsafe_allow_html=True)

            if truth:
                pred_pos = p >= threshold
                match = (truth == "pneumonia") == pred_pos
                if verdict.key == "borderline":
                    outcome = "result is borderline"
                else:
                    outcome = "prediction matches" if match else "prediction differs"
                st.markdown(
                    h(f'<span class="pill">Dataset label: {truth.title()} · {outcome}</span>'),
                    unsafe_allow_html=True,
                )

# --------------------------------------------------------------------------------------
# About / limitations
# --------------------------------------------------------------------------------------
st.markdown("---")
tab_about, tab_limits = st.tabs(["About the models", "Limitations"])

with tab_about:
    st.markdown(
        """
**How it works.** Three ImageNet-pretrained CNNs (ResNet-50, DenseNet-121, EfficientNet-B0) were fine-tuned
to classify chest X-rays as *normal* or *pneumonia*. Each image is resized to 224×224 and scored by every model.
The raw outputs are **temperature-scaled** on a held-out validation set so that probabilities are less over-confident,
and the ensemble averages them. Train/validation splits were grouped by patient to avoid leakage.
        """
    )
    metrics = cfg.get("test_metrics_with_ci")
    if metrics:
        label_of = {**DISPLAY_NAMES, "ensemble": "Ensemble"}
        body = "".join(
            f"<tr><td><b>{label_of.get(m, m)}</b></td><td>{v.get('threshold', '')}</td><td>{v.get('auc', '')}</td>"
            f"<td>{v.get('sensitivity', '')}</td><td>{v.get('specificity', '')}</td><td>{v.get('accuracy', '')}</td></tr>"
            for m, v in metrics.items()
        )
        st.markdown(h(f"""
        <table class="perf"><thead><tr><th>Model</th><th>Threshold</th><th>AUC</th><th>Sensitivity</th>
        <th>Specificity</th><th>Accuracy</th></tr></thead><tbody>{body}</tbody></table>
        """), unsafe_allow_html=True)
        st.caption(
            "Official Kaggle test split (624 images). Brackets are 95% patient-level bootstrap confidence intervals. "
            "The table uses operating points chosen on a validation set; the slider in the sidebar lets you try any threshold."
        )

with tab_limits:
    st.markdown(
        """
- **Single dataset.** Trained and tested on one public, paediatric chest X-ray collection. There is no external validation,
  and results may not carry over to adults, other scanners or other hospitals.
- **Many false alarms.** Sensitivity is very high, but specificity is only around 60–70% on the test split, so a lot of
  healthy X-rays are flagged as pneumonia.
- **Possible shortcuts.** In a masking experiment, image regions outside the central chest still carried enough signal to
  classify, so the models may rely partly on non-anatomical cues (positioning, markers, borders).
- **No input validation.** The app only checks for obvious colour photos. It cannot confirm that an image is a chest X-ray.
- **Not clinical software.** There is no regulatory clearance, and the output must not guide medical decisions.
        """
    )

st.markdown('<div class="footer">Built with PyTorch and Streamlit · Research demo, not for clinical use</div>', unsafe_allow_html=True)
