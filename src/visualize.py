from __future__ import annotations

import cv2
import numpy as np

SHAPE_COLORS = {
    "Cuboid": (255, 80, 80),
    "Cylindrical": (80, 160, 255),
    "Flat": (80, 220, 80),
    "Irregular": (255, 200, 0),
}
_DEFAULT_COLOR = (200, 200, 200)


def visualize_predictions(result: dict) -> np.ndarray:
    img = result["image"].copy()
    for obj in result.get("objects", []):
        if "shape" not in obj:
            continue
        x1, y1, x2, y2 = (int(v) for v in obj["bbox"])
        color = SHAPE_COLORS.get(obj["shape"], _DEFAULT_COLOR)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f'{obj["shape"]} {obj["confidence"]:.0%}'
        y_text = max(15, y1 - 8)
        cv2.putText(img, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img


def colorize_depth(depth_map: np.ndarray) -> np.ndarray:
    d = depth_map.astype(np.float32)
    span = d.max() - d.min()
    d = (d - d.min()) / span if span > 1e-6 else np.zeros_like(d)
    return cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
