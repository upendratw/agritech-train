from pathlib import Path

LBL_DIR = Path("/Users/upendra/Downloads/Jowar.v1i.yolov8/train/labels")

fixed = 0
deleted = 0

for lbl in LBL_DIR.glob("*.txt"):
    valid_lines = []

    with open(lbl) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                valid_lines.append(" ".join(parts))

    if not valid_lines:
        lbl.unlink()
        deleted += 1
    else:
        with open(lbl, "w") as f:
            f.write("\n".join(valid_lines) + "\n")
        fixed += 1

print(f"Fixed labels: {fixed}")
print(f"Deleted empty labels: {deleted}")