import cv2
import matplotlib.pyplot as plt

# ---- PATHS ----
image_path = r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\stage1_dataset\images\train\doc_00001.png"
label_path = r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\stage1_dataset\labels\train\doc_00001.txt"

# ---- READ IMAGE ----
img = cv2.imread(image_path)
h, w = img.shape[:2]

# ---- READ LABELS ----
boxes = []
with open(label_path, "r") as f:
    for line in f:
        cls, cx, cy, bw, bh = map(float, line.strip().split())
        cls = int(cls)

        # Convert YOLO normalized → pixel bounding box
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)

        boxes.append([cls, x1, y1, x2, y2])

# ---- DRAW BOXES ----
for cls, x1, y1, x2, y2 in boxes:
    color = (0, 255, 0) if cls == 0 else (255, 0, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, str(cls), (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# ---- SHOW IMAGE ----
plt.figure(figsize=(10, 14))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
