# Desk Phone Detector: Small Evaluation Study

This project uses a pretrained YOLOv8s COCO model to detect phones in desk images. The live webcam script applies a confidence cutoff and plays an alarm when a phone is detected. `tunethreshold.py` is the controlled evaluation for comparing confidence cutoffs.

## Question

Does raising the phone confidence cutoff from 0.40 reduce false alarms while retaining useful phone detections on images held out from threshold selection?

## Dataset and limitations

The checked-in Roboflow export contains 77 images, each with a YOLO label file. The export has one directory at `Dataset/train`; its `val` and `test` entries both point to that same directory. Those entries are not independent evaluation splits, so the old validation figures are not a valid held-out result and are removed here. The images were captured in one short session, so even a deterministic image-level holdout may contain near-duplicate views. Treat this as an exploratory mini-study, not evidence of performance in other rooms or lighting.

Labels are phone bounding boxes. Empty label files mean no phone is annotated. The detector is COCO class 67 (`cell phone`); dataset labels are treated as the single phone class.

## Evaluation protocol

The script makes a reproducible 20% image-level holdout using seed 42 and evaluates a pretrained model without training. It runs inference once at confidence 0.20, then filters the same detections at 0.20, 0.40, 0.60, 0.70, and 0.80. A prediction is a true positive if its box has IoU >= 0.50 with an unmatched ground-truth phone. Predictions and ground-truth boxes are matched greedily in descending prediction confidence; duplicate detections count as false positives. Unmatched ground-truth phones count as false negatives.

Precision = TP / (TP + FP). Recall = TP / (TP + FN). F1 is their harmonic mean. If a denominator is zero, its metric is reported as zero. The best cutoff is selected by F1 among these five tested values; this does not establish a globally optimal threshold.

Run from the project folder:

```powershell
python tunethreshold.py
```

Optional arguments: `--model PATH` selects weights, and `--holdout FRACTION` changes the holdout size. The command prints TP, FP, FN, precision, recall, and F1 for every cutoff. It also saves side-by-side comparable baseline (0.40) and selected-cutoff overlays in `Results/threshold_examples/`. Inspect these images for examples where the threshold removed a false alarm and examples where it removed a real phone detection; thresholding cannot improve localization and may reduce recall.

## Interpreting the comparison

Latest run (15 held-out images; seed 42):

| Confidence | TP | FP | FN | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.20 | 8 | 7 | 5 | 0.533 | 0.615 | **0.571** |
| 0.40 baseline | 7 | 7 | 6 | 0.500 | 0.538 | 0.519 |
| 0.60 | 6 | 5 | 7 | 0.545 | 0.462 | 0.500 |
| 0.70 | 5 | 4 | 8 | 0.556 | 0.385 | 0.455 |
| 0.80 | 3 | 2 | 10 | 0.600 | 0.231 | 0.333 |

For this holdout, 0.20 raised F1 from 0.519 to 0.571 compared with 0.40, with recall increasing by 7.7 percentage points while precision increased by 3.3 points. The 0.80 cutoff gave the highest precision but missed two thirds of the labeled phones. The live detector is set to the best tested F1 cutoff, 0.20. This small sample gives a noisy estimate and is not independent of the same capture session. Do not reuse previously quoted aggregate values: they came from overlapping train/val/test paths and are not held-out evidence.

## Live detector

`detection.py` opens the webcam, detects COCO phones, and loops `alarm.wav` while a phone is present. Its live cutoff is 0.20, the best tested F1 setting on the holdout. The live script uses Windows `winsound`.
