#!/usr/bin/env python3
"""
coco2yolo.py

Convert a COCO annotation JSON (images + annotations + categories) into YOLO-style
label files. For each image referenced in the JSON, a text file with same name
and extension `.txt` will be created next to the image (same folder) containing
one line per object:

<class_idx> <x_center> <y_center> <width> <height>

Coordinates are normalized to [0,1] relative to image width/height.

Usage:
  python coco2yolo.py --coco /Users/upendra/Downloads/dataset_final/annotations/train.json \
                      --images_root /Users/upendra/Downloads/dataset_final/images \
                      --out_labels_root /Users/upendra/Downloads/dataset_final/labels_train

If --out_labels_root is omitted, .txt files are written next to image file (mirrors images structure).
"""
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
from PIL import Image

def load_coco(coco_json_path):
    with open(coco_json_path, "r") as f:
        j = json.load(f)
    images = {img["id"]: img for img in j.get("images", [])}
    anns = j.get("annotations", [])
    cats = j.get("categories", [])
    # category id -> continuous index (0..N-1)
    cat_id_to_idx = {}
    for i, c in enumerate(sorted(cats, key=lambda x: x["id"])):
        cat_id_to_idx[c["id"]] = i
    return images, anns, cat_id_to_idx, cats

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def convert_annotations(coco_json_path, images_root, out_labels_root=None, require_images=True, force_image_size_check=False):
    images, anns, cat_id_to_idx, cats = load_coco(coco_json_path)
    # group annotations by image_id
    by_img = defaultdict(list)
    for a in anns:
        by_img[a["image_id"]].append(a)

    images_root = Path(images_root)
    if out_labels_root:
        out_labels_root = Path(out_labels_root)

    created = 0
    missing_images = []
    for img_id, img_meta in images.items():
        file_name = img_meta["file_name"]
        img_path = images_root / file_name
        if not img_path.exists():
            # try alternative path: check absolute path in file_name
            if Path(file_name).exists():
                img_path = Path(file_name)
            else:
                missing_images.append(str(img_path))
                continue

        # get actual width/height
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception:
            # fallback to metadata in JSON
            W = img_meta.get("width")
            H = img_meta.get("height")
            if not W or not H:
                print(f"[WARN] Cannot open image and no size in JSON: {img_path}")
                continue

        # determine label path
        label_rel = Path(file_name).with_suffix(".txt")
        if out_labels_root:
            label_path = out_labels_root / label_rel
        else:
            label_path = images_root / label_rel

        ensure_dir(label_path.parent)

        ann_list = by_img.get(img_id, [])
        lines = []
        for a in ann_list:
            cat_id = a["category_id"]
            if cat_id not in cat_id_to_idx:
                # skip unknown category
                continue
            cls_idx = cat_id_to_idx[cat_id]
            bbox = a.get("bbox")  # COCO bbox: [x, y, w, h]
            if not bbox:
                continue
            x, y, w, h = bbox
            # convert to x_center,y_center, normalized
            x_c = x + w / 2.0
            y_c = y + h / 2.0
            x_c_n = x_c / W
            y_c_n = y_c / H
            w_n = w / W
            h_n = h / H
            # clamp
            def clamp(v):
                return max(0.0, min(1.0, float(v)))
            lines.append(f"{cls_idx} {clamp(x_c_n):.6f} {clamp(y_c_n):.6f} {clamp(w_n):.6f} {clamp(h_n):.6f}")

        # write label file (overwrite)
        with open(label_path, "w") as fo:
            fo.write("\n".join(lines))
        created += 1

    print(f"Processed {created} images. Missing images: {len(missing_images)}")
    if missing_images:
        print("Examples of missing images (first 10):")
        for p in missing_images[:10]:
            print("  ", p)

    # build list of category names in order of continuous indices
    cat_names = [None] * (max(cat_id_to_idx.values()) + 1)
    for cat in cats:
        idx = cat_id_to_idx[cat["id"]]
        cat_names[idx] = cat.get("name", str(cat["id"]))

    return cat_names, cat_id_to_idx

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--coco", required=True, help="COCO json file path")
    p.add_argument("--images_root", required=True, help="Root folder where image paths in COCO are relative to")
    p.add_argument("--out_labels_root", default=None, help="Optional folder to write labels into (mirrors file_name folders). If omitted, labels are written next to images.")
    p.add_argument("--require_images", action="store_true", help="If set, fail when image missing (default only warns).")
    args = p.parse_args()

    cats, mapping = convert_annotations(args.coco, args.images_root, out_labels_root=args.out_labels_root)
    print("Categories (index order):")
    for i, n in enumerate(cats):
        print(i, n)