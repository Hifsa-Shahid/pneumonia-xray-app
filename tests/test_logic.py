import numpy as np
from PIL import Image

from src.logic import agreement, decide, image_checks


def test_decide_regions():
    assert decide(0.90, 0.50).key == "pneumonia"
    assert decide(0.05, 0.50).key == "normal"
    assert decide(0.55, 0.50).key == "borderline"
    assert decide(0.40, 0.50).key == "borderline"


def test_decide_follows_threshold():
    assert decide(0.30, 0.10).key == "pneumonia"
    assert decide(0.30, 0.90).key == "normal"


def test_agreement():
    assert agreement({"a": 0.9, "b": 0.2, "c": 0.7}, 0.5) == (2, 3)
    assert agreement({"a": 0.5}, 0.5) == (1, 1)


def test_grayscale_image_is_ok():
    g = Image.fromarray((np.random.rand(256, 256) * 255).astype(np.uint8)).convert("RGB")
    warnings, blocking = image_checks(g)
    assert not blocking and not warnings


def test_colour_photo_is_blocked():
    rgb = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))
    _, blocking = image_checks(rgb)
    assert blocking


def test_tiny_image_warns():
    tiny = Image.new("RGB", (64, 64), (128, 128, 128))
    warnings, blocking = image_checks(tiny)
    assert warnings and not blocking
