# convert_to_yolo.py
import os, json, cv2, random, shutil
from tqdm import tqdm

# Mapping: category_id in JSON → YOLO class index
category_map = {
    1: "Text",
    2: "Title",
    3: "List",
    4: "Table",
    5: "Figure"
}

def convert_dataset(src_dir="data\english_Dataset", out_dir="data\stage1_dataset", val_split=0.1):
    # Create output directory structure
    img_train = os.path.join(out_dir, "images/train")
    img_val = os.path.join(out_dir, "images/val")
    lbl_train = os.path.join(out_dir, "labels/train")
    lbl_val = os.path.join(out_dir, "labels/val")

    for d in [img_train, img_val, lbl_train, lbl_val]:
        os.makedirs(d, exist_ok=True)

    # Get all images
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(files)
    n_val = int(len(files) * val_split)

    for i, img_name in enumerate(tqdm(files, desc="Converting to YOLO format")):
        img_path = os.path.join(src_dir, img_name)
        json_path = os.path.join(src_dir, img_name.rsplit('.', 1)[0] + ".json")

        if not os.path.exists(json_path):
            continue

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        with open(json_path, "r") as f:
            data = json.load(f)

        lines = []
        for ann in data.get("annotations", []):
            cat_id = ann["category_id"]
            if cat_id not in category_map:
                continue

            x, y, bw, bh = ann["bbox"]

            # Convert to YOLO normalized format
            x_c = (x + bw / 2) / w
            y_c = (y + bh / 2) / h
            bw /= w
            bh /= h

            # YOLO format: class x_center y_center width height
            lines.append(f"{cat_id-1} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

        # Train/Val Split
        if i < n_val:
            img_out, lbl_out = img_val, lbl_val
        else:
            img_out, lbl_out = img_train, lbl_train

        shutil.copy(img_path, os.path.join(img_out, img_name))

        label_file = img_name.rsplit('.', 1)[0] + ".txt"
        with open(os.path.join(lbl_out, label_file), "w") as f:
            f.write("\n".join(lines))

    print(f"\n✅ YOLO dataset (Stage 1) created successfully at: {out_dir}")


if __name__ == "__main__":
    convert_dataset(src_dir="data\english_Dataset", out_dir="data\stage1_dataset")
