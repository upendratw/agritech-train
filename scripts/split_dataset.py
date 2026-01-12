#!/usr/bin/env python3
"""
split_dataset.py

If you already have YOLO images and labels in a flat folder, split them into train/val.

Usage:
  python split_dataset.py --images_dir data/yolo/images_all --labels_dir data/yolo/labels_all --out data/yolo --train_frac 0.8
"""
import argparse
from pathlib import Path
import random
import shutil
from tqdm import tqdm

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--labels_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    img_dir = Path(args.images_dir)
    lbl_dir = Path(args.labels_dir)
    out_root = Path(args.out)

    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png"]])
    random.seed(args.seed)
    random.shuffle(imgs)
    split = int(len(imgs) * args.train_frac)
    train = imgs[:split]
    val = imgs[split:]

    for subset, lst in [("train", train), ("val", val)]:
        out_img = out_root / "images" / subset
        out_lbl = out_root / "labels" / subset
        ensure_dir(out_img); ensure_dir(out_lbl)
        for src in tqdm(lst, desc=f"Copying {subset}"):
            dest_img = out_img / src.name
            shutil.copy2(src, dest_img)
            lbl_src = Path(lbl_dir) / (src.stem + ".txt")
            if lbl_src.exists():
                shutil.copy2(lbl_src, out_lbl / lbl_src.name)
            else:
                # create empty label
                (out_lbl / (src.stem + ".txt")).write_text("")

    print("Done. train/val sizes:", len(train), len(val))

if __name__ == "__main__":
    main()