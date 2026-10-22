"""Model loading and CPU inference for the three pneumonia classifiers."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_NAMES = ("resnet50", "densenet121", "efficientnet_b0")
DISPLAY_NAMES = {
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0",
}
DEFAULT_CONFIG = {
    "class_names": ["NORMAL", "PNEUMONIA"],
    "img_size": 224,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "models": {n: {"checkpoint": f"models/{n}.pt", "temperature": 1.0} for n in MODEL_NAMES},
}

torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))


def load_config() -> dict:
    """Read models/app_config.json (written by the training notebook); fall back to safe defaults."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    path = MODEL_DIR / "app_config.json"
    if path.exists():
        user = json.loads(path.read_text())
        for key, value in user.items():
            if key == "models":
                for name, entry in value.items():
                    cfg["models"].setdefault(name, {}).update(entry)
            else:
                cfg[key] = value
    return cfg


def build_model(name: str) -> nn.Module:
    """Same architecture/heads as the training notebook (no pretrained download needed)."""
    if name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 2))
    elif name == "densenet121":
        m = models.densenet121(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier.in_features, 2))
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, 2))
    else:
        raise ValueError(f"Unknown model: {name}")
    return m


def _checkpoint_path(cfg: dict, name: str) -> Path:
    return ROOT / cfg["models"][name]["checkpoint"]


class Engine:
    """Lazy, thread-safe holder of the models. Create once and cache it in Streamlit."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._models: dict[str, nn.Module] = {}
        self._lock = threading.Lock()
        size = int(cfg["img_size"])
        self._tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(cfg["mean"], cfg["std"]),
        ])

    def available(self) -> list[str]:
        return [n for n in MODEL_NAMES if _checkpoint_path(self.cfg, n).exists()]

    def _get(self, name: str) -> nn.Module:
        with self._lock:
            if name not in self._models:
                state = torch.load(_checkpoint_path(self.cfg, name), map_location="cpu", weights_only=True)
                state = {k: (v.float() if v.is_floating_point() else v) for k, v in state.items()}  # fp16 -> fp32
                model = build_model(name)
                model.load_state_dict(state)
                self._models[name] = model.eval()
            return self._models[name]

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict[str, float]:
        """Calibrated P(pneumonia) per available model: softmax(logits / T)[1]."""
        x = self._tf(image.convert("RGB")).unsqueeze(0)
        out: dict[str, float] = {}
        for name in self.available():
            logits = self._get(name)(x).float()[0].numpy().astype(np.float64)
            temp = float(self.cfg["models"][name].get("temperature", 1.0))
            z = logits / temp
            z = z - z.max()
            e = np.exp(z)
            out[name] = float(e[1] / e.sum())
        return out
