"""
server_v2.py — Upgraded FastAPI backend
- Auto-detects fruit type using 26-class classifier (no dropdown)
- Multi-item detection: finds individual items in batch photos
- Uses contour detection + YOLOv8 combined for robust item finding
- Each item gets its own freshness + shelf life analysis

Run: uvicorn api.server_v2:app --host 0.0.0.0 --port 8000 --reload
"""
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import io
import os
import sqlite3
from datetime import datetime

import sys
sys.path.insert(0, os.path.expanduser("~/freshscan"))

from engine.decision_engine_v2 import decide_action, analyze_multi_items
from engine.llm_explainer_v2 import (
    generate_explanation,
    generate_multi_item_explanation,
    generate_daily_report,
)

app = FastAPI(title="FreshScan AI v2", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.expanduser("~/freshscan/models/best_classifier_v2.pth")
DB_PATH = os.path.expanduser("~/freshscan/freshscan_logs.db")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classifier = None
class_names = []
yolo_model = None


def parse_class_name(class_name):
    """Convert 'fresh_apple' → ('apple', 'fresh')"""
    parts = class_name.split("_", 1)
    if len(parts) == 2:
        return parts[1], parts[0]
    return class_name, "unknown"


def load_models():
    global classifier, class_names, yolo_model

    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Classifier not found at {MODEL_PATH}")
        return False

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    num_classes = checkpoint.get('num_classes', len(checkpoint.get('classes', [])))
    class_names = checkpoint.get('classes', [])

    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()
    classifier = model
    print(f"Classifier loaded! {num_classes} classes: {class_names}")

    try:
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n.pt")
        print("YOLOv8 loaded for object detection!")
    except Exception as e:
        print(f"YOLOv8 not available: {e}")
        print("Will use contour-based detection instead.")

    return True


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, fruit_type TEXT, freshness TEXT,
            confidence REAL, shelf_life_hours REAL, action TEXT,
            discount_pct INTEGER, explanation TEXT, multi_item INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

init_db()
load_models()


def classify_crop(image_pil):
    """Run the 26-class classifier on a PIL image crop."""
    tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = classifier(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top_idx = probs.argmax().item()
    top_class = class_names[top_idx]
    top_conf = float(probs[top_idx])
    fruit_type, freshness = parse_class_name(top_class)

    top5_vals, top5_idxs = probs.topk(min(5, len(class_names)))
    top5 = {class_names[int(i)]: round(float(v), 4) for v, i in zip(top5_vals, top5_idxs)}

    return {
        "predicted_class": top_class,
        "fruit_type": fruit_type,
        "freshness": freshness,
        "confidence": top_conf,
        "top5_predictions": top5,
    }


# ============================================
# Multi-Item Detection: Contour-Based
# ============================================
def find_items_contour(image_pil, min_area_ratio=0.01, max_area_ratio=0.80):
    """
    Find individual produce items using OpenCV contour detection.
    Works for ANY fruit/vegetable — doesn't need YOLO to recognize the type.

    Strategy:
    1. Convert to HSV color space
    2. Create mask for non-background pixels (assumes lighter/white-ish background)
    3. Find contours (individual items)
    4. Filter by size
    5. Return bounding boxes
    """
    img_np = np.array(image_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    total_area = h * w
    min_area = total_area * min_area_ratio
    max_area = total_area * max_area_ratio

    # Method 1: HSV-based segmentation (works well for colorful produce)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Create mask: anything that isn't very light (background) or very dark
    lower = np.array([0, 30, 30])
    upper = np.array([180, 255, 255])
    mask_hsv = cv2.inRange(hsv, lower, upper)

    # Method 2: Edge-based detection as backup
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_edge = cv2.dilate(edges, kernel, iterations=3)
    mask_edge = cv2.morphologyEx(mask_edge, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Combine both masks
    mask = cv2.bitwise_or(mask_hsv, mask_edge)

    # Clean up
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_clean, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_clean, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # Filter out very thin/weird shapes
        aspect = bw / max(bh, 1)
        if aspect > 5 or aspect < 0.2:
            continue

        # Add padding (10% of box size)
        pad_x = int(bw * 0.10)
        pad_y = int(bh * 0.10)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        boxes.append([x1, y1, x2, y2])

    # Remove overlapping boxes (non-maximum suppression)
    boxes = nms_boxes(boxes, iou_threshold=0.3)

    return boxes


def nms_boxes(boxes, iou_threshold=0.3):
    """Simple non-maximum suppression to remove overlapping detections."""
    if len(boxes) <= 1:
        return boxes

    boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    keep = []

    for i, box_a in enumerate(boxes):
        should_keep = True
        for box_b in keep:
            iou = compute_iou(box_a, box_b)
            if iou > iou_threshold:
                should_keep = False
                break
        if should_keep:
            keep.append(box_a)

    return keep


def compute_iou(box_a, box_b):
    """Compute intersection over union of two boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0


def find_items_yolo(image_pil):
    """Use YOLOv8 to find objects. Returns bounding boxes."""
    if yolo_model is None:
        return []

    try:
        results = yolo_model(image_pil, conf=0.25, verbose=False)
        boxes_out = []
        w, h = image_pil.size

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            pad = 15
            x1 = max(0, int(x1) - pad)
            y1 = max(0, int(y1) - pad)
            x2 = min(w, int(x2) + pad)
            y2 = min(h, int(y2) + pad)

            # Only keep reasonably sized detections
            box_area = (x2 - x1) * (y2 - y1)
            if box_area > 400:  # at least 20x20 pixels
                boxes_out.append([x1, y1, x2, y2])

        return boxes_out
    except Exception as e:
        print(f"YOLO error: {e}")
        return []


def find_all_items(image_pil):
    """
    Combined detection strategy:
    1. Try YOLO first (good for common items like bananas, apples, oranges)
    2. Also run contour detection (catches okra, bittergourd, capsicum etc.)
    3. Merge results, remove duplicates with NMS
    """
    yolo_boxes = find_items_yolo(image_pil)
    contour_boxes = find_items_contour(image_pil)

    # Combine
    all_boxes = yolo_boxes + contour_boxes

    # Remove duplicates
    all_boxes = nms_boxes(all_boxes, iou_threshold=0.4)

    return all_boxes


# ============================================
# Endpoints
# ============================================

@app.post("/scan")
async def scan_produce(file: UploadFile = File(...)):
    """
    Smart multi-item scan.
    Finds all items in the image, classifies each one.
    """
    if classifier is None:
        return JSONResponse(status_code=503,
            content={"error": "Model not loaded."})

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Find all items
    boxes = find_all_items(image)

    detected_items = []

    if len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = box
            crop = image.crop((x1, y1, x2, y2))

            # Skip tiny crops
            if crop.size[0] < 30 or crop.size[1] < 30:
                continue

            classification = classify_crop(crop)

            # Only keep high-confidence detections
            if classification["confidence"] > 0.40:
                classification["bbox"] = [x1, y1, x2, y2]
                detected_items.append(classification)

    # If nothing detected with boxes, classify the whole image
    if not detected_items:
        classification = classify_crop(image)
        classification["bbox"] = None
        detected_items.append(classification)

    # Build decisions for each item
    item_decisions = []
    for item in detected_items:
        decision = decide_action(item["fruit_type"], item["freshness"], item["confidence"])
        decision["bbox"] = item.get("bbox")
        decision["top5_predictions"] = item.get("top5_predictions", {})
        item_decisions.append(decision)

    # Single vs multi response
    if len(item_decisions) == 1:
        result = item_decisions[0]
        result["explanation"] = generate_explanation(result)
        result["multi_item"] = False
        log_scan(result)
        return result
    else:
        multi_input = [
            {"fruit_type": d["fruit_type"], "freshness": d["freshness"],
             "confidence": d["confidence"]}
            for d in item_decisions
        ]
        summary = analyze_multi_items(multi_input)

        for i, item in enumerate(summary["items"]):
            item["bbox"] = item_decisions[i].get("bbox")
            item["top5_predictions"] = item_decisions[i].get("top5_predictions", {})

        summary["explanation"] = generate_multi_item_explanation(summary)
        summary["multi_item"] = True

        for item in summary["items"]:
            log_scan(item)

        return summary


@app.post("/scan-single")
async def scan_single(file: UploadFile = File(...)):
    """
    Single-item scan — classifies the whole image as one item.
    Fastest option, no detection needed.
    """
    if classifier is None:
        return JSONResponse(status_code=503,
            content={"error": "Model not loaded."})

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    classification = classify_crop(image)
    decision = decide_action(
        classification["fruit_type"],
        classification["freshness"],
        classification["confidence"]
    )
    decision["top5_predictions"] = classification["top5_predictions"]
    decision["explanation"] = generate_explanation(decision)
    decision["multi_item"] = False
    decision["bbox"] = None

    log_scan(decision)
    return decision


def log_scan(result):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """INSERT INTO scans
               (timestamp, fruit_type, freshness, confidence,
                shelf_life_hours, action, discount_pct, explanation, multi_item)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (result.get("timestamp", datetime.now().isoformat()),
             result["fruit_type"], result["freshness"], result["confidence"],
             result["shelf_life_hours"], result["action"],
             result["discount_percentage"],
             result.get("explanation", ""),
             1 if result.get("multi_item") else 0)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB error: {e}")


@app.get("/stats")
async def get_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("""
        SELECT action, COUNT(*), AVG(confidence), AVG(shelf_life_hours)
        FROM scans WHERE DATE(timestamp) = DATE('now')
        GROUP BY action
    """)
    stats = {}
    for row in cursor:
        stats[row[0]] = {
            "count": row[1],
            "avg_confidence": round(row[2], 3),
            "avg_shelf_life": round(row[3], 1),
        }

    total_cursor = conn.execute("""
        SELECT COUNT(*), SUM(CASE WHEN action='compost' THEN 1 ELSE 0 END)
        FROM scans WHERE DATE(timestamp) = DATE('now')
    """)
    row = total_cursor.fetchone()
    total = row[0] or 0
    composted = row[1] or 0
    conn.close()

    return {
        "by_action": stats,
        "total_scans": total,
        "waste_rate": round(composted / total * 100, 1) if total > 0 else 0,
    }


@app.get("/daily-report")
async def daily_report():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("""
        SELECT fruit_type, freshness, action, confidence, shelf_life_hours
        FROM scans WHERE DATE(timestamp) = DATE('now')
    """)
    scans = [{"fruit_type": r[0], "freshness": r[1], "action": r[2],
              "confidence": r[3], "shelf_life_hours": r[4]} for r in cursor]
    conn.close()

    if not scans:
        return {"report": "No scans recorded today.", "total_scans": 0}

    report = generate_daily_report(scans)
    return {"report": report, "total_scans": len(scans)}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "classifier_loaded": classifier is not None,
        "yolo_loaded": yolo_model is not None,
        "num_classes": len(class_names),
        "classes": class_names,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cuda": torch.cuda.is_available(),
    }
