from __future__ import annotations

import os

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_WEIGHTS = os.environ.get(
    "YOLO_BASELINE_WEIGHTS", os.path.join(_THIS_DIR, "..", "models", "yolov8s_960_swarnim_baseline.pt")
)

_yolo_model = None


def run_yolo_baseline(image: np.ndarray, conf_threshold: float = 0.25) -> np.ndarray:
    global _yolo_model
    if not os.path.exists(YOLO_WEIGHTS):
        raise FileNotFoundError(f"YOLO baseline checkpoint not found at {YOLO_WEIGHTS}")
    if _yolo_model is None:
        from ultralytics import YOLO

        _yolo_model = YOLO(YOLO_WEIGHTS)

    result = _yolo_model(image, imgsz=960, conf=conf_threshold, verbose=False)[0]
    return result.plot()[:, :, ::-1]


if __name__ == "__main__":
    import argparse

    from PIL import Image as PILImage

    parser = argparse.ArgumentParser(description="Run Swarnim's YOLOv8s baseline on one image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default=None, help="Output path (default: <image>_yolo_baseline.png next to input)")
    args = parser.parse_args()

    img = np.array(PILImage.open(args.image).convert("RGB"))
    annotated = run_yolo_baseline(img)
    out_path = args.out or os.path.splitext(args.image)[0] + "_yolo_baseline.png"
    PILImage.fromarray(annotated).save(out_path)
    print(f"Saved {out_path}")
