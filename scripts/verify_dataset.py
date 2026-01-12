#!/usr/bin/env python3
"""
verify_dataset.py

Quick checks on YOLO dataset:
 - counts images and labels
 - prints class distribution
 - finds images without label files and label files without images

Usage:
  python verify_dataset.py --yolo_root data/yolo
"""
import argparse
from pathlib import Path
from collections import Counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_root", required=True)
    args = parser.parse_args()

    root = Path(args.yolo_root)
    for split in ["train", "val"]:
        imgs_dir = root / "images" / split
        lbls_dir = root / "labels" / split
        if not imgs_dir.exists():
            print(f"Missing {imgs_dir} — skipping.")
            continue
        img_files = sorted([p for p in imgs_dir.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png"]])
        lbl_files = sorted([p for p in lbls_dir.iterdir() if p.suffix.lower() == ".txt"]) if lbls_dir.exists() else []
        print(f"=== {split} ===")
        print("images:", len(img_files), "labels:", len(lbl_files))
        missing_labels = []
        for p in img_files:
            if not (lbls_dir / (p.stem + ".txt")).exists():
                missing_labels.append(p.name)
        if missing_labels:
            print("Images without labels (sample 20):", missing_labels[:20])
        orphan_labels = []
        for p in lbl_files:
            if not (imgs_dir / (p.stem + ".jpg")).exists() and not (imgs_dir / (p.stem + ".png")).exists():
                orphan_labels.append(p.name)
        if orphan_labels:
            print("Label files without images (sample 20):", orphan_labels[:20])

        # class distribution
        class_counts = Counter()
        for p in lbl_files:
            with open(p, "r") as fh:
                for line in fh:
                    parts = line.strip().split()
                    if not parts: continue
                    class_counts[int(parts[0])] += 1
        print("Class counts (label lines):", dict(class_counts))
        print()

if __name__ == "__main__":
    main()