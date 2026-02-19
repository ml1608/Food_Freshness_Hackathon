"""
server_v3.py — Final FreshScan AI Backend

Workflow:
1. Vendor selects fruit type from dropdown (e.g., "tomato")
2. Uploads a batch photo of multiple items
3. System detects each individual item using contour detection
4. Classifies each as fresh/rotten using the 98% accuracy 2-class model
5. Returns per-item results + batch summary with average shelf life

Run: uvicorn api.server_v3:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw
import cv2
import numpy as np
import io
import os
import sqlite3
import base64
from datetime import datetime

import sys
sys.path.insert(0, os.path.expanduser("~/freshscan"))

from engine.decision_engine_v2 import decide_action, analyze_multi_items
from engine.llm_explainer_v2 import (
    generate_explanation,
    generate_multi_item_explanation,
    generate_daily_report,
)

app = FastAPI(title="FreshScan AI v3", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.expanduser("~/freshscan/models/best_classifier.pth")  # Original 2-class model
DB_PATH = os.path.expanduser("~/freshscan/freshscan_logs.db")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classifier = None
class_names = ["fresh", "rotten"]


def load_model():
    global classifier, class_names

    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model not found at {MODEL_PATH}")
        return False

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 2)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()
    classifier = model

    if 'classes' in checkpoint:
        class_names = checkpoint['classes']

    print(f"Classifier loaded! Classes: {class_names}")
    print(f"Using 2-class model (98% accuracy) at: {MODEL_PATH}")
    return True


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, fruit_type TEXT, freshness TEXT,
            confidence REAL, shelf_life_hours REAL, action TEXT,
            discount_pct INTEGER, explanation TEXT,
            batch_id TEXT, item_number INTEGER
        )
    """)
    conn.commit()
    conn.close()

init_db()
load_model()


def classify_freshness(image_pil):
    """
    Classify a single image crop as fresh or rotten.
    Returns freshness label and confidence.
    """
    tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = classifier(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    fresh_idx = class_names.index("fresh") if "fresh" in class_names else 0
    rotten_idx = class_names.index("rotten") if "rotten" in class_names else 1

    top_idx = probs.argmax().item()
    freshness = class_names[top_idx]
    confidence = float(probs[top_idx])

    return {
        "freshness": freshness,
        "confidence": confidence,
        "fresh_prob": round(float(probs[fresh_idx]), 4),
        "rotten_prob": round(float(probs[rotten_idx]), 4),
    }


# ============================================
# Item Detection using OpenCV
# ============================================
def find_individual_items(image_pil, min_area_ratio=0.008, max_area_ratio=0.85):
    """
    Find individual produce items in a batch photo using contour detection.
    Designed for photos of multiple items of the same type on a surface.
    """
    img_np = np.array(image_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    total_area = h * w
    min_area = total_area * min_area_ratio
    max_area = total_area * max_area_ratio

    # Convert to HSV for color-based segmentation
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Mask: keep saturated, non-white, non-black pixels (i.e., the produce)
    lower = np.array([0, 25, 25])
    upper = np.array([180, 255, 255])
    mask_color = cv2.inRange(hsv, lower, upper)

    # Also use Otsu thresholding on grayscale for contrast-based detection
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, mask_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Edge detection
    edges = cv2.Canny(blurred, 25, 80)
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mask_edge = cv2.dilate(edges, kernel_edge, iterations=3)

    # Combine all masks
    mask = cv2.bitwise_or(mask_color, mask_otsu)
    mask = cv2.bitwise_or(mask, mask_edge)

    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Watershed-like separation for touching items
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.35 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Find contours on the cleaned mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # Filter extreme aspect ratios
        aspect = bw / max(bh, 1)
        if aspect > 6 or aspect < 0.16:
            continue

        # Add 12% padding
        pad_x = int(bw * 0.12)
        pad_y = int(bh * 0.12)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        boxes.append({
            "bbox": [x1, y1, x2, y2],
            "area": area,
        })

    # Sort by area (largest first) and apply NMS
    boxes.sort(key=lambda b: b["area"], reverse=True)
    filtered = nms_boxes([b["bbox"] for b in boxes], iou_threshold=0.35)

    return filtered


def nms_boxes(boxes, iou_threshold=0.35):
    if len(boxes) <= 1:
        return boxes

    keep = []
    for box_a in boxes:
        should_keep = True
        for box_b in keep:
            if compute_iou(box_a, box_b) > iou_threshold:
                should_keep = False
                break
        if should_keep:
            keep.append(box_a)
    return keep


def compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def draw_results_on_image(image_pil, item_results):
    """Draw bounding boxes and labels on the image for visualization."""
    draw = ImageDraw.Draw(image_pil)

    for i, item in enumerate(item_results):
        bbox = item.get("bbox")
        if not bbox:
            continue

        x1, y1, x2, y2 = bbox
        freshness = item["freshness"]
        confidence = item["confidence"]

        # Green for fresh, red for rotten
        color = (0, 200, 0) if freshness == "fresh" else (220, 0, 0)

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label
        label = f"#{i+1} {freshness.upper()} {confidence:.0%}"
        # Background for text
        draw.rectangle([x1, y1 - 22, x1 + len(label) * 8, y1], fill=color)
        draw.text((x1 + 4, y1 - 20), label, fill=(255, 255, 255))

    return image_pil


# ============================================
# Endpoints
# ============================================

@app.post("/scan-batch")
async def scan_batch(
    file: UploadFile = File(...),
    fruit_type: str = Form(...)
):
    """
    Batch scan: vendor selects fruit type, uploads photo of multiple items.
    System detects each item and classifies fresh/rotten individually.
    """
    if classifier is None:
        return JSONResponse(status_code=503,
            content={"error": "Model not loaded."})

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Find individual items
    boxes = find_individual_items(image)

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    item_results = []

    if len(boxes) > 0:
        for idx, bbox in enumerate(boxes):
            x1, y1, x2, y2 = bbox
            crop = image.crop((x1, y1, x2, y2))

            if crop.size[0] < 25 or crop.size[1] < 25:
                continue

            freshness_result = classify_freshness(crop)

            decision = decide_action(
                fruit_type,
                freshness_result["freshness"],
                freshness_result["confidence"]
            )
            decision["bbox"] = bbox
            decision["item_number"] = idx + 1
            decision["fresh_prob"] = freshness_result["fresh_prob"]
            decision["rotten_prob"] = freshness_result["rotten_prob"]

            item_results.append(decision)

    # If no items detected, classify whole image
    if not item_results:
        freshness_result = classify_freshness(image)
        decision = decide_action(
            fruit_type,
            freshness_result["freshness"],
            freshness_result["confidence"]
        )
        decision["bbox"] = None
        decision["item_number"] = 1
        decision["fresh_prob"] = freshness_result["fresh_prob"]
        decision["rotten_prob"] = freshness_result["rotten_prob"]
        item_results.append(decision)

    # Build annotated image
    annotated = image.copy()
    annotated = draw_results_on_image(annotated, item_results)
    buffered = io.BytesIO()
    annotated.save(buffered, format="JPEG", quality=90)
    annotated_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Aggregate summary
    fresh_items = [r for r in item_results if r["freshness"] == "fresh"]
    rotten_items = [r for r in item_results if r["freshness"] == "rotten"]

    avg_shelf_fresh = 0
    if fresh_items:
        avg_shelf_fresh = sum(r["shelf_life_hours"] for r in fresh_items) / len(fresh_items)

    avg_conf = sum(r["confidence"] for r in item_results) / len(item_results)

    summary = {
        "batch_id": batch_id,
        "fruit_type": fruit_type,
        "total_items": len(item_results),
        "fresh_count": len(fresh_items),
        "rotten_count": len(rotten_items),
        "average_shelf_life_hours": round(avg_shelf_fresh, 1),
        "average_shelf_life_display": (
            f"{avg_shelf_fresh/24:.1f} days" if avg_shelf_fresh >= 48
            else f"{avg_shelf_fresh:.0f} hours"
        ),
        "average_confidence": round(avg_conf, 3),
        "items": item_results,
        "annotated_image": annotated_b64,
    }

    # Generate LLM explanation
    multi_input = [
        {"fruit_type": fruit_type, "freshness": r["freshness"], "confidence": r["confidence"]}
        for r in item_results
    ]
    multi_summary = analyze_multi_items(multi_input)
    summary["explanation"] = generate_multi_item_explanation(multi_summary)

    # Recommended batch action
    if len(rotten_items) == 0:
        summary["batch_recommendation"] = f"All {len(item_results)} {fruit_type}(s) are fresh. Sell at full price."
    elif len(fresh_items) == 0:
        summary["batch_recommendation"] = f"All {len(item_results)} {fruit_type}(s) are rotten. Remove and compost immediately."
    else:
        summary["batch_recommendation"] = (
            f"Mixed batch: {len(fresh_items)} fresh, {len(rotten_items)} rotten. "
            f"Remove rotten items (#{', #'.join(str(r['item_number']) for r in rotten_items)}). "
            f"Fresh items have avg {summary['average_shelf_life_display']} shelf life."
        )

    # Log each item
    for item in item_results:
        log_scan(item, batch_id)

    return summary


@app.post("/scan-single")
async def scan_single(
    file: UploadFile = File(...),
    fruit_type: str = Form(...)
):
    """Single item scan with fruit type from dropdown."""
    if classifier is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded."})

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    freshness_result = classify_freshness(image)
    decision = decide_action(fruit_type, freshness_result["freshness"], freshness_result["confidence"])
    decision["fresh_prob"] = freshness_result["fresh_prob"]
    decision["rotten_prob"] = freshness_result["rotten_prob"]
    decision["top5_predictions"] = {
        "fresh": freshness_result["fresh_prob"],
        "rotten": freshness_result["rotten_prob"],
    }
    decision["explanation"] = generate_explanation(decision)
    decision["multi_item"] = False

    log_scan(decision, None)
    return decision


def log_scan(result, batch_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """INSERT INTO scans
               (timestamp, fruit_type, freshness, confidence,
                shelf_life_hours, action, discount_pct, explanation,
                batch_id, item_number)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (result.get("timestamp", datetime.now().isoformat()),
             result["fruit_type"], result["freshness"], result["confidence"],
             result["shelf_life_hours"], result["action"],
             result["discount_percentage"],
             result.get("explanation", ""),
             batch_id,
             result.get("item_number", 0))
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
        "version": "3.0.0",
        "classifier_loaded": classifier is not None,
        "model_type": "2-class (fresh/rotten) — 98% accuracy",
        "num_classes": len(class_names),
        "classes": class_names,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cuda": torch.cuda.is_available(),
    }
