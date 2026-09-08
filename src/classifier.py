from __future__ import annotations

import os
import random
import sys
from typing import Protocol

import numpy as np

GEOMETRY_CLASSES = ["Flat", "Cylindrical", "Cuboid", "Irregular"]

_PRANAYA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pranaya_classifier")


class GeometryClassifier(Protocol):
    def predict(self, rgb_crop: np.ndarray, depth_crop: np.ndarray, mask: np.ndarray | None = None) -> dict:
        ...


class MockGeometryClassifier:
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def predict(self, rgb_crop: np.ndarray, depth_crop: np.ndarray, mask: np.ndarray | None = None) -> dict:
        return {
            "label": self._rng.choice(GEOMETRY_CLASSES),
            "confidence": round(self._rng.uniform(0.55, 0.95), 2),
        }


class PranayaClassifier:
    def __init__(self, mode: str = "real") -> None:
        if mode not in ("real", "synthetic"):
            raise ValueError(f"unknown Pranaya classifier mode: {mode!r} (expected 'real' or 'synthetic')")
        if _PRANAYA_DIR not in sys.path:
            sys.path.insert(0, _PRANAYA_DIR)

        if mode == "real":
            from real_classifier import RealClassifier

            self._impl = RealClassifier()
        else:
            from synthetic_classifier import SyntheticClassifier

            self._impl = SyntheticClassifier()
        self.mode = mode

    def predict(self, rgb_crop: np.ndarray, depth_crop: np.ndarray, mask: np.ndarray | None = None) -> dict:
        result = self._impl.predict(rgb_crop, depth_crop=depth_crop, mask=mask)
        return {"label": result["class"], "confidence": result["confidence"]}


def build_geometry_classifier(model_path: str | None = None) -> GeometryClassifier:
    model_path = model_path or os.environ.get("GEOMETRY_CLASSIFIER_CHECKPOINT")
    if model_path is None:
        return MockGeometryClassifier()

    if model_path in ("real", "synthetic"):
        return PranayaClassifier(mode=model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"geometry classifier checkpoint not found at {model_path}")

    mode = "synthetic" if "synthetic" in os.path.basename(model_path).lower() else "real"
    return PranayaClassifier(mode=mode)
