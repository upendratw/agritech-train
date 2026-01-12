#!/usr/bin/env python3
"""
make_data_yaml.py

Generates a data.yaml file for Ultralytics YOLO training.

Usage:
  python make_data_yaml.py \
    --train_images /Users/upendra/Downloads/dataset_final/images/train \
    --val_images /Users/upendra/Downloads/dataset_final/images/val \
    --test_images /Users/upendra/Downloads/dataset_final/images/test \
    --train_json /Users/upendra/Downloads/dataset_final/annotations/train.json \
    --out data.yaml
"""
import argparse
import json
from pathlib import Path

def extract_names_from_coco(json_path):
    with open(json_path, "r") as f:
        j = json.load(f)
    cats = j.get("categories", [])
    # sort categories by id and produce names list in index order 0..N-1
    cats_sorted = sorted(cats, key=lambda c: c["id"])
    names = [c.get("name", str(c["id"])) for c in cats_sorted]
    return names

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_images", required=True)
    p.add_argument("--val_images", required=True)
    p.add_argument("--test_images", default=None)
    p.add_argument("--train_json", required=True, help="COCO train json (to extract categories)")
    p.add_argument("--out", default="data.yaml")
    args = p.parse_args()

    names = extract_names_from_coco(args.train_json)
    out = {
        "train": str(Path(args.train_images).resolve()),
        "val": str(Path(args.val_images).resolve())
    }
    if args.test_images:
        out["test"] = str(Path(args.test_images).resolve())
    out["nc"] = len(names)
    out["names"] = names

    out_path = Path(args.out)
    out_path.write_text(
        "\n".join([
            f"train: {out['train']}",
            f"val: {out['val']}",
            f"test: {out.get('test', '')}" if out.get("test") else "",
            f"nc: {out['nc']}",
            "names:",
            *[f"  {i}: {n}" for i, n in enumerate(names)]
        ])
    )
    print(f"Wrote {out_path.resolve()}")