#!/usr/bin/env python3
"""
visualize_yolo.py

Show an image with YOLO labels drawn (boxes + class id). Useful to verify conversion.

Usage:
  python visualize_yolo.py --image data/yolo/images/train/img1.jpg --label data/yolo/labels/train/img1.txt --names data/yolo/data.names
"""
import argparse
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def read_labels(path):
    out = []
    if not Path(path).exists():
        return out
    with open(path, "r") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            out.append((cls, xc, yc, w, h))
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--names", required=False)
    parser.add_argument("--out", required=False, help="If provided, save to file instead of show")
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        from PIL import ImageFont
        font = ImageFont.load_default()

    labels = read_labels(args.label)
    names = []
    if args.names:
        with open(args.names, "r") as fh:
            names = [l.strip() for l in fh.readlines() if l.strip()]

    for cls, xc, yc, w, h in labels:
        x1 = int((xc - w/2.0) * W)
        y1 = int((yc - h/2.0) * H)
        x2 = int((xc + w/2.0) * W)
        y2 = int((yc + h/2.0) * H)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        txt = str(cls) if not names else (names[cls] if cls < len(names) else str(cls))
        draw.text((x1+3, y1+3), txt, fill="white", font=font)

    if args.out:
        img.save(args.out)
        print("Saved:", args.out)
    else:
        img.show()

if __name__ == "__main__":
    main()