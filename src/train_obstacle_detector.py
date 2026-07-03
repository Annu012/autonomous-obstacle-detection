"""Train YOLOv11 for Autonomous Vehicle Obstacle Detection"""

from ultralytics import YOLO
import torch
import yaml
from pathlib import Path
import json

class ObstacleDetectorTrainer:
    def __init__(self, model_size: str = "m"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
    
    def prepare_config(self):
        config = {
            "path": str(Path("data").absolute()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": 4,
            "names": {0: "car", 1: "person", 2: "cyclist", 3: "truck"}
        }
        
        Path("config").mkdir(exist_ok=True)
        with open("config/obstacle_detection.yaml", 'w') as f:
            yaml.dump(config, f)
        
        return "config/obstacle_detection.yaml"
    
    def train(self, epochs: int = 100, batch_size: int = 16):
        config = self.prepare_config()
        model = YOLO(f"yolo11{self.model_size}.pt")
        
        print(f"Training YOLO11-{self.model_size} for Obstacle Detection")
        results = model.train(
            data=config,
            epochs=epochs,
            imgsz=1280,
            batch=batch_size,
            device=self.device,
            patience=25,
            save=True,
            plots=True,
            project="runs/detect",
            name="train"
        )
        return results
    
    def evaluate_safety(self, model_path: str) -> dict:
        model = YOLO(model_path)
        metrics = model.val()
        
        fnr = 1 - float(metrics.box.mr)
        
        return {
            "mAP50": float(metrics.box.map50),
            "mAP": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "false_negative_rate": fnr,
            "safe": fnr < 0.05
        }

if __name__ == "__main__":
    trainer = ObstacleDetectorTrainer(model_size="m")
    results = trainer.train(epochs=100, batch_size=16)
    metrics = trainer.evaluate_safety("runs/detect/train/weights/best.pt")
    
    print("\n=== SAFETY EVALUATION ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    
    Path("results").mkdir(exist_ok=True)
    with open("results/obstacle_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

