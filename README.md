# Desk Phone Detector & Confidence Threshold Study

A real-time Computer Vision system using YOLOv8 to detect smartphones on a desk surface and trigger audio alerts, backed by a controlled benchmark study on confidence threshold optimization.


## Executive Summary

Deploying pre-trained object detectors off-the-shelf often leads to false positives caused by domain-specific clutter (e.g., computer mice, power banks, wallets, calculators). Rather than retraining the network, this study evaluates the zero-shot performance of `YOLOv8s` on a custom-annotated desk environment dataset and applies empirical confidence threshold tuning to balance precision and recall.

---

## Problem Definition & Evaluation Criteria
* **Goal:** Detect a smartphone on a desk surface with high reliability while minimizing false alarms.
* **True Positive (TP):** Predicted bounding box overlaps a ground-truth smartphone with IoU >= 0.5.
* **False Positive (FP):** Non-phone object detected as a phone, or box overlap IoU < 0.5.
* **False Negative (FN):** Visible smartphone missed by detector or below confidence cutoff.

---

## Benchmark Dataset
* **Sample Size:** 77 annotated images of desk environments under varying lighting, angles, and clutter conditions.
* **Annotation Format:** YOLOv8 bounding boxes via Roboflow.
* **Target Class:** Smartphone / Cell Phone.

---

## Experimental Results

### Confidence Sweep Table

| Confidence Cutoff (`conf`) | Precision | Recall | F1-Score | Remarks |
| :--- | :--- | :--- | :--- | :--- |
| **0.20** | 0.2776 | **0.4865** | 0.3535 | High false alarm rate on clutter |
| **0.40 (Baseline)** | 0.2776 | **0.4865** | 0.3535 | Original script baseline |
| **0.60** | 0.2812 | **0.4865** | 0.3564 | Moderate noise reduction |
| **0.70 (Optimal)** | **0.3197** | 0.4054 | **0.3575** | **Optimal F1 balance point** |
| **0.80** | **0.4146** | 0.2393 | 0.3035 | High precision, severe recall drop |

---

## Key Takeaways
1. **Baseline Flaws:** Default confidence settings (`0.20`–`0.40`) yield poor precision (27.8%) due to background noise triggering low-confidence predictions.
2. **Optimal Setting:** Tuning `conf` to **0.70** yields the maximum F1-Score (0.3575), effectively eliminating background noise while retaining target coverage.
3. **Precision-Recall Trade-off:** Increasing `conf` to `0.80` raises precision to 41.5%, but recall drops by 50.8%, causing frequent missed detections.

---

## Project Structure
```text
├── Photos/              # Test images folder
├── .gitignore           # Excludes heavy run folders and model weights
├── alarm.wav            # Audio trigger alert
├── detection.py         # Live webcam detector with alarm integration
├── evaluate.py          # Baseline evaluation script
└── tunethreshold.py     # Multi-threshold sweep script