from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Load stock YOLOv8 model
    model = YOLO('yolov8s.pt')
    
    # 2. Run validation against your dataset
    results = model.val(
        data='Dataset/data.yaml', 
        conf=0.4, 
        single_cls=True
    )
    
    print("\n--- Baseline Evaluation Complete ---")
    print(f"Precision: {results.results_dict['metrics/precision(B)']:.4f}")
    print(f"Recall:    {results.results_dict['metrics/recall(B)']:.4f}")
    print(f"mAP50:     {results.results_dict['metrics/mAP50(B)']:.4f}")
    print("Check 'runs/detect/val' for generated graphs!")