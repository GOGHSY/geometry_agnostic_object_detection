from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import numpy as np

from classifier import build_geometry_classifier
from pipeline import run_pipeline
from visualize import colorize_depth, visualize_predictions
from yolo_baseline import run_yolo_baseline

_MODEL_CHOICES = {
    "Real-Trained (Pranaya)": "real",
    "Synthetic-Trained (Pranaya, sim-to-real)": "synthetic",
}
_classifier_cache: dict[str | None, object] = {}


def _get_classifier(choice_label: str):
    mode = _MODEL_CHOICES[choice_label]
    if mode not in _classifier_cache:
        _classifier_cache[mode] = build_geometry_classifier(mode)
    return _classifier_cache[mode]


def _get_yolo_baseline(image: np.ndarray):
    try:
        return run_yolo_baseline(image)
    except Exception:
        return None


def infer(image: np.ndarray | None, model_choice: str):
    if image is None:
        return None, None, None, "Upload an image first."

    yolo_vis = _get_yolo_baseline(image)

    try:
        classifier = _get_classifier(model_choice)
        result = run_pipeline(image, classifier=classifier)
    except Exception as e:
        return None, None, yolo_vis, f"Pipeline error: {e}"

    if result.get("error"):
        return None, None, yolo_vis, f"Pipeline error: {result['error']}"
    if result["depth_map"] is None:
        return None, None, yolo_vis, "Pipeline produced no depth map."
    if not result["objects"]:
        return image, colorize_depth(result["depth_map"])[:, :, ::-1], yolo_vis, "No objects detected."

    annotated = visualize_predictions(result)
    depth_vis = colorize_depth(result["depth_map"])[:, :, ::-1]
    lines = []
    for obj in result["objects"]:
        if "shape" in obj:
            lines.append(f"#{obj['object_id']}: {obj['shape']} ({obj['confidence']:.0%})")
        else:
            lines.append(f"#{obj['object_id']}: failed ({obj.get('error')})")
    return annotated, depth_vis, yolo_vis, "\n".join(lines)


demo = gr.Interface(
    fn=infer,
    inputs=[
        gr.Image(label="Input Image", type="numpy"),
        gr.Radio(
            list(_MODEL_CHOICES.keys()),
            value="Real-Trained (Pranaya)",
            label="Geometry classifier",
        ),
    ],
    outputs=[
        gr.Image(label="Two-Stage Pipeline: shape + confidence"),
        gr.Image(label="Depth Map"),
        gr.Image(label="RGB Baseline: YOLOv8s (Swarnim, class-agnostic 'item')"),
        gr.Textbox(label="Per-object predictions"),
    ],
    title="Class-Agnostic Geometric Detection",
    description=(
        "MobileSAM proposals -> Depth Anything V2 -> geometry classifier, next to "
        "Swarnim's frozen YOLOv8s baseline for comparison. First run of a "
        "Real/Synthetic classifier takes ~15-30s to load, then it's cached."
    ),
)

if __name__ == "__main__":
    demo.launch()
