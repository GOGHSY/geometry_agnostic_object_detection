from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image as PILImage

from classifier import GeometryClassifier, build_geometry_classifier

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(_THIS_DIR, "..", "models"))
MOBILE_SAM_WEIGHTS = os.path.join(MODELS_DIR, "mobile_sam.pt")
DEPTH_MODEL_ID = os.environ.get("DEPTH_MODEL_ID", "depth-anything/Depth-Anything-V2-Small-hf")

_sam_model = None
_depth_pipe = None


def generate_proposals(
    image: np.ndarray,
    conf_threshold: float = 0.0,
    iou_threshold: float = 0.6,
    max_detections: int = 50,
) -> list[dict]:
    global _sam_model
    if not os.path.exists(MOBILE_SAM_WEIGHTS):
        raise FileNotFoundError(
            f"mobile_sam.pt not found at {MOBILE_SAM_WEIGHTS} "
            "(auto-downloads on first run with network access, or drop it in manually)"
        )
    if _sam_model is None:
        from ultralytics import SAM

        _sam_model = SAM(MOBILE_SAM_WEIGHTS)

    result = _sam_model(image, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.tolist()
    scores = result.boxes.conf.tolist()
    boxes_wh = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]

    keep = cv2.dnn.NMSBoxes(boxes_wh, scores, score_threshold=conf_threshold, nms_threshold=iou_threshold)
    keep = keep.flatten().tolist() if len(keep) else []
    keep.sort(key=lambda i: -scores[i])
    keep = keep[:max_detections]

    masks_np = result.masks.data[keep].cpu().numpy().astype(bool) if result.masks is not None else None

    proposals = [
        {
            "bbox": [round(v, 1) for v in boxes[i]],
            "score": float(scores[i]),
            "mask": masks_np[j] if masks_np is not None else None,
        }
        for j, i in enumerate(keep)
    ]
    return proposals


def generate_depth(image: np.ndarray) -> np.ndarray:
    global _depth_pipe
    if _depth_pipe is None:
        from transformers import pipeline as hf_pipeline

        _depth_pipe = hf_pipeline(task="depth-estimation", model=DEPTH_MODEL_ID)

    h, w = image.shape[:2]
    out = _depth_pipe(PILImage.fromarray(image))
    depth = np.array(out["depth"], dtype=np.float32)
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    return depth


def expand_bbox(bbox: list[float], image_width: int, image_height: int, padding_ratio: float = 0.15) -> list[float]:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = w * padding_ratio, h * padding_ratio
    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(image_width, x2 + pad_x),
        min(image_height, y2 + pad_y),
    ]


def run_pipeline(
    image: np.ndarray,
    classifier: GeometryClassifier | None = None,
    padding_ratio: float = 0.15,
) -> dict:
    classifier = classifier or build_geometry_classifier()
    h, w = image.shape[:2]

    try:
        proposals = generate_proposals(image)
    except Exception as e:
        return {"image": image, "depth_map": None, "objects": [], "error": f"proposal generation failed: {e}"}

    try:
        depth_map = generate_depth(image)
    except Exception as e:
        return {"image": image, "depth_map": None, "objects": [], "error": f"depth generation failed: {e}"}

    objects = []
    for i, prop in enumerate(proposals):
        try:
            bbox = prop["bbox"]
            px1, py1, px2, py2 = (int(round(v)) for v in expand_bbox(bbox, w, h, padding_ratio))
            if px2 <= px1 or py2 <= py1:
                raise ValueError("degenerate padded bbox")
            rgb_crop = image[py1:py2, px1:px2]
            depth_crop = depth_map[py1:py2, px1:px2]
            mask_full = prop.get("mask")
            mask_crop = mask_full[py1:py2, px1:px2] if mask_full is not None else None
            pred = classifier.predict(rgb_crop, depth_crop, mask=mask_crop)
            objects.append(
                {
                    "object_id": i,
                    "bbox": bbox,
                    "score": prop["score"],
                    "padded_bbox": [px1, py1, px2, py2],
                    "rgb_crop": rgb_crop,
                    "depth_crop": depth_crop,
                    "shape": pred["label"],
                    "confidence": pred["confidence"],
                }
            )
        except Exception as e:
            objects.append({"object_id": i, "bbox": prop.get("bbox"), "error": str(e)})

    return {"image": image, "depth_map": depth_map, "objects": objects}


def _cli() -> None:
    import argparse

    from visualize import colorize_depth, visualize_predictions

    parser = argparse.ArgumentParser(description="Run the class-agnostic geometry pipeline on one image.")
    parser.add_argument("--image", required=True, help="Path to an RGB image.")
    parser.add_argument("--out-dir", default=os.path.join(_THIS_DIR, "..", "results", "figures"))
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"ERROR: input image not found: {args.image}")
        return

    image = np.array(PILImage.open(args.image).convert("RGB"))
    result = run_pipeline(image)
    if result.get("error"):
        print(f"ERROR: {result['error']}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    PILImage.fromarray(visualize_predictions(result)).save(os.path.join(args.out_dir, f"{stem}_annotated.png"))
    PILImage.fromarray(colorize_depth(result["depth_map"])[:, :, ::-1]).save(
        os.path.join(args.out_dir, f"{stem}_depth.png")
    )

    print(f"{len(result['objects'])} object(s) detected in {args.image}:")
    for obj in result["objects"]:
        if "shape" in obj:
            print(f"  #{obj['object_id']}: {obj['shape']} (conf={obj['confidence']:.2f}) bbox={obj['bbox']}")
        else:
            print(f"  #{obj['object_id']}: FAILED - {obj.get('error')}")
    print(f"Saved visualizations to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    _cli()
