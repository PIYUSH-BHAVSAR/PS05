import os
import json
from collections import defaultdict

def count_categories(folder_path):
    print(f"Scanning folder: {folder_path}\n")

    category_counts = defaultdict(int)   # Stores total category counts
    total_files = 0
    json_files = 0

    # Walk through all subfolders and files
    for root, _, files in os.walk(folder_path):
        for file in files:
            total_files += 1
            if file.endswith(".json"):
                json_files += 1
                json_path = os.path.join(root, file)

                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    annotations = data.get("annotations", [])

                    for ann in annotations:
                        cid = ann.get("category_id")
                        category_counts[cid] += 1

                except Exception as e:
                    print(f"Error reading {json_path}: {e}")

    # ---- PRINT SUMMARY ----
    print("\n========== CATEGORY SUMMARY ==========")
    for cid in sorted(category_counts.keys()):
        print(f"Category {cid}: {category_counts[cid]} annotations")

    print("\nTotal JSON files scanned:", json_files)
    print("Total files scanned (all types):", total_files)
    print("=====================================\n")


if __name__ == "__main__":
    folder = r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\train"
    count_categories(folder)
