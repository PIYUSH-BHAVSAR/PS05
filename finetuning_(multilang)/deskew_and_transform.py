# deskew_and_transform.py
import cv2, os, json, math, numpy as np
from glob import glob
from tqdm import tqdm

def estimate_skew_angle(img_gray):
    edges = cv2.Canny(img_gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 120)  # lowered threshold for better stability
    if lines is None: 
        return 0.0
    angles = []
    for l in lines[:200]:
        rho, theta = l[0]
        angle = (theta - np.pi/2) * (180/np.pi)
        angles.append(angle)
    if len(angles) == 0:
        return 0.0
    return np.median(angles)

def rotate_image_and_boxes(img, bboxes, angle):
    h,w = img.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0,0]); sin = abs(M[0,1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0,2] += (nW/2) - center[0]
    M[1,2] += (nH/2) - center[1]

    rotated = cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR, borderValue=(255,255,255))

    new_boxes = []
    for box in bboxes:
        x,y,wb,hb = box
        pts = np.array([[x,y],[x+wb,y],[x+wb,y+hb],[x,y+hb]], dtype=np.float32)
        ones = np.ones((pts.shape[0],1))
        pts_h = np.hstack([pts, ones])
        transformed = (M.dot(pts_h.T)).T
        x_min = float(np.min(transformed[:,0]))
        y_min = float(np.min(transformed[:,1]))
        x_max = float(np.max(transformed[:,0]))
        y_max = float(np.max(transformed[:,1]))
        new_boxes.append([x_min, y_min, x_max-x_min, y_max-y_min])
    return rotated, new_boxes

def process_folder(img_dir, ann_dir, out_img_dir, out_ann_dir):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_ann_dir, exist_ok=True)

    img_paths = sorted(glob(os.path.join(img_dir, "*.png")) + glob(os.path.join(img_dir, "*.jpg")))

    for img_path in tqdm(img_paths):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        ann_path = os.path.join(ann_dir, basename + ".json")
        if not os.path.exists(ann_path):
            continue

        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ✅ main fix: reverse sign so deskew is correct
        angle = -estimate_skew_angle(img_gray)

        with open(ann_path, 'r') as f:
            ann = json.load(f)

        if abs(angle) < 0.5:
            out_img = img
            new_boxes = [obj['bbox'] for obj in ann.get('annotations', [])]
        else:
            bboxes = [obj['bbox'] for obj in ann.get('annotations', [])]
            out_img, new_boxes = rotate_image_and_boxes(img, bboxes, angle)

        out_img_path = os.path.join(out_img_dir, basename + ".png")
        cv2.imwrite(out_img_path, out_img)

        new_ann = {'file_name': os.path.basename(out_img_path), 'annotations': []}
        for idx, obj in enumerate(ann.get('annotations', [])):
            cat = obj.get('category_id', None)
            bbox = new_boxes[idx]
            new_ann['annotations'].append({'category_id': cat, 'bbox': [float(round(x,2)) for x in bbox]})

        with open(os.path.join(out_ann_dir, basename + ".json"), 'w') as f:
            json.dump(new_ann, f, indent=2)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--img_dir', required=True)
    p.add_argument('--ann_dir', required=True)
    p.add_argument('--out_img_dir', required=True)
    p.add_argument('--out_ann_dir', required=True)
    args = p.parse_args()
    process_folder(args.img_dir, args.ann_dir, args.out_img_dir, args.out_ann_dir)
