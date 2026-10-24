"""Evaluate confidence cutoffs on a deterministic, held-out image split.

The exported dataset currently contains one 77-image directory. This script
splits it by image (not by detection) and evaluates one shared set of model
predictions at several confidence thresholds. Matching is greedy, one-to-one,
and uses IoU >= 0.5.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "yolov8s.pt"
SPLIT_SEED = 42
IOU_THRESHOLD = 0.5
CONFIDENCE_CUTOFFS = (0.20, 0.40, 0.60, 0.70, 0.80)


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def dataset_paths() -> tuple[list[Path], Path]:
    config_path = ROOT / "Dataset" / "data.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_root = ROOT / config.get("path", ".")
    image_dir = data_root / config["train"]
    label_dir = image_dir.parent / "labels"
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return images, label_dir


def read_labels(path: Path, width: int, height: int) -> list[list[float]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Expected YOLO class/x/y/w/h in {path}: {line!r}")
        _, xc, yc, bw, bh = map(float, fields)
        boxes.append([(xc - bw / 2) * width, (yc - bh / 2) * height,
                      (xc + bw / 2) * width, (yc + bh / 2) * height])
    return boxes


def evaluate_at(predictions, cutoff: float) -> dict[str, float | int]:
    tp = fp = fn = 0
    for sample in predictions:
        candidates = [p for p in sample["predictions"] if p["confidence"] >= cutoff]
        candidates.sort(key=lambda p: p["confidence"], reverse=True)
        matched: set[int] = set()
        for pred in candidates:
            overlaps = [(box_iou(pred["box"], gt), idx)
                        for idx, gt in enumerate(sample["ground_truth"])
                        if idx not in matched]
            best_iou, best_idx = max(overlaps, default=(0.0, -1))
            if best_iou >= IOU_THRESHOLD:
                tp += 1
                matched.add(best_idx)
            else:
                fp += 1
        fn += len(sample["ground_truth"]) - len(matched)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--holdout", type=float, default=0.20)
    args = parser.parse_args()
    if not 0 < args.holdout < 1:
        parser.error("--holdout must be between 0 and 1")

    images, labels_dir = dataset_paths()
    shuffled = images.copy()
    random.Random(SPLIT_SEED).shuffle(shuffled)
    holdout_count = max(1, round(len(images) * args.holdout))
    holdout = sorted(shuffled[:holdout_count])
    model = YOLO(str(args.model))
    predictions = []

    # Use a low floor once, then filter these same detections for every cutoff.
    for result in model.predict(source=[str(p) for p in holdout], classes=[67],
                                conf=min(CONFIDENCE_CUTOFFS), iou=IOU_THRESHOLD,
                                stream=False, verbose=False):
        height, width = result.orig_shape
        gt = read_labels(labels_dir / f"{Path(result.path).stem}.txt", width, height)
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                detections.append({"box": box.xyxy[0].tolist(), "confidence": float(box.conf[0])})
        predictions.append({"ground_truth": gt, "predictions": detections})

    print(f"Held-out images: {len(holdout)}/{len(images)} (seed={SPLIT_SEED}); IoU match >= {IOU_THRESHOLD:.2f}")
    print(f"{'cutoff':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'precision':>11} {'recall':>9} {'F1':>8}")
    results = []
    for cutoff in CONFIDENCE_CUTOFFS:
        metrics = evaluate_at(predictions, cutoff)
        results.append((cutoff, metrics))
        print(f"{cutoff:8.2f} {metrics['tp']:6d} {metrics['fp']:6d} {metrics['fn']:6d} "
              f"{metrics['precision']:11.3f} {metrics['recall']:9.3f} {metrics['f1']:8.3f}")
    best_cutoff, best_metrics = max(results, key=lambda item: item[1]["f1"])
    print(f"Best tested cutoff by held-out F1: {best_cutoff:.2f} ({best_metrics['f1']:.3f})")

    example_dir = ROOT / "Results" / "threshold_examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    for image_path, sample in zip(holdout, predictions):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        for cutoff, label in ((0.40, "baseline_040"), (best_cutoff, f"selected_{best_cutoff:.2f}")):
            frame = image.copy()
            for det in sample["predictions"]:
                if det["confidence"] < cutoff:
                    continue
                x1, y1, x2, y2 = map(int, det["box"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 210, 60), 2)
                cv2.putText(frame, f"phone {det['confidence']:.2f}", (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 210, 60), 2)
            cv2.imwrite(str(example_dir / f"{image_path.stem}_{label}.jpg"), frame)
    print(f"Saved baseline/selected-cutoff visual examples to {example_dir}")


if __name__ == "__main__":
    main()
