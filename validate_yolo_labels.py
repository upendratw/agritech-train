from pathlib import Path
from PIL import Image

IMG_DIR = Path("/Users/upendra/Downloads/uraddalV1-dataset/train/images")
LBL_DIR = Path("/Users/upendra/Downloads/uraddalV1-dataset/train/labels")

bad = []

for img_path in IMG_DIR.glob("*"):
    lbl_path = LBL_DIR / (img_path.stem + ".txt")
    if not lbl_path.exists():
        continue

    img = Image.open(img_path)
    W, H = img.size

    with open(lbl_path) as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) != 5:
                bad.append((img_path, "invalid format"))
                continue

            cls, x, y, w, h = map(float, parts)

            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                bad.append((img_path, "out of range"))

            if w == 0 or h == 0:
                bad.append((img_path, "zero box"))

print("Bad files:", len(bad))
for b in bad[:20]:
    print(b)