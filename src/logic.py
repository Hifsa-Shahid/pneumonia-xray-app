"""Pure-Python decision logic and image sanity checks (no torch import, easy to unit-test)."""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

BAND = 0.10  # +/- distance around the threshold that counts as "borderline"


@dataclass(frozen=True)
class Verdict:
    key: str      # "pneumonia" | "normal" | "borderline"
    title: str
    detail: str


def decide(p: float, threshold: float, band: float = BAND) -> Verdict:
    """Turn a calibrated P(pneumonia) into a human-readable screening result."""
    if abs(p - threshold) <= band:
        return Verdict(
            "borderline", "Borderline result",
            "The probability is close to the decision threshold, so the model is not confident either way.",
        )
    if p > threshold:
        return Verdict(
            "pneumonia", "Pneumonia likely",
            "The estimated probability is above your decision threshold.",
        )
    return Verdict(
        "normal", "Pneumonia unlikely",
        "The estimated probability is below your decision threshold.",
    )


def agreement(probs: dict[str, float], threshold: float) -> tuple[int, int]:
    """Return (number of models at/above the threshold, number of models)."""
    n_pos = sum(1 for p in probs.values() if p >= threshold)
    return n_pos, len(probs)


def load_image(data: bytes) -> Image.Image:
    """Decode bytes into an 8-bit RGB image (handles 16-bit grayscale PNGs)."""
    img = Image.open(io.BytesIO(data))
    img.load()
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(img, dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
        img = Image.fromarray(arr.astype(np.uint8))
    return img.convert("RGB")


def image_checks(img: Image.Image) -> tuple[list[str], bool]:
    """Cheap sanity checks. Returns (warnings, blocking).

    `blocking` is True when the image is clearly not a greyscale radiograph
    (e.g. a colour photo), because the model would output a meaningless score.
    """
    warnings: list[str] = []
    blocking = False
    w, h = img.size
    if min(w, h) < 128:
        warnings.append("The image is very small, so the result may be unreliable.")
    ratio = w / max(h, 1)
    if ratio > 2.0 or ratio < 0.5:
        warnings.append("The image has an unusual shape for a chest X-ray.")
    small = np.asarray(img.convert("RGB").resize((96, 96)), dtype=np.float32)
    chroma = float((np.abs(small[..., 0] - small[..., 1]) + np.abs(small[..., 1] - small[..., 2])).mean() / 2)
    if chroma > 12:
        warnings.append("This looks like a colour photo, not a greyscale chest X-ray.")
        blocking = True
    return warnings, blocking
