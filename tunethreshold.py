from ultralytics import YOLO
import numpy as np

if __name__ == '__main__':
    model = YOLO('yolov8s.pt')
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    
    print("CONFIDENCE THRESHOLD SWEEP RESULTS: ")
    print(f"{'Conf Threshold':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 55)
    
    best_f1 = 0
    best_conf = 0.4

    for conf in thresholds:
        results = model.val(
            data='Dataset/data.yaml', 
            conf=conf, 
            single_cls=True,
            verbose=False
        )
        
        p = results.results_dict['metrics/precision(B)']
        r = results.results_dict['metrics/recall(B)']
        
        # Calculate F1 score (harmonic mean of Precision & Recall)
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        
        print(f"{conf:<16.2f} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_conf = conf

    print("-" * 55)
    print(f"Optimal Threshold: conf = {best_conf:.2f} (Max F1-Score: {best_f1:.4f})")