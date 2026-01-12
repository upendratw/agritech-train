#!/usr/bin/env python3
"""
training/train.py

Python training wrapper using ultralytics YOLO (yolov8).
Adjust args as needed or call from shell.

Usage:
  python training/train.py --data ../configs/data.yaml --model yolov8n.pt --imgsz 512 --epochs 200
"""
import argparse
from ultralytics import YOLO
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to data yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model or path")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu or mps or cuda")
    parser.add_argument("--project", default="./results")
    parser.add_argument("--name", default="agritech_model")
    args = parser.parse_args()

    os.makedirs(args.project, exist_ok=True)
    print("Training with:", args)
    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    print("Training finished. Results under:", os.path.join(args.project, args.name))

if __name__ == "__main__":
    main()