"""
server.py — FreshScan AI Backend (Final)

Uses SwinV2-Tiny for texture-aware freshness classification.
All logic in one file: detection, classification, decisions, LLM, daily report.

Run: uvicorn api.server:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw
import timm
import io
import os
import sqlite3
import base64
import requests as http_requests
from datetime import datetime

app = FastAPI(title="FreshScan AI", version="4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.expanduser("~/freshscan/models/best_classifier.pth")
DB_PATH = os.path.expanduser("~/freshscan/freshscan_logs.db")

# ══════════════════════════════════════════════════════════════
# GLOBALS
# ══════════════════════════════════════════════════════════════
classifier = None
class_names = ["fresh", "rotten"]
img_size = 256
yolo = None
model_name = "swinv2_tiny_window8_256"

# Transform must match training
img_transform = None

def build_transform(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# ══════════════════════════════════════════════════════════════
# SHELF LIFE TABLE (USDA FoodKeeper)
# ══════════════════════════════════════════════════════════════
SHELF_LIFE = {
    "apple":       {"max_hours": 336, "note": "Up to 2 weeks refrigerated"},
    "banana":      {"max_hours": 168, "note": "5-7 days at room temp"},
    "orange":      {"max_hours": 504, "note": "Up to 3 weeks refrigerated"},
    "tomato":      {"max_hours": 168, "note": "5-7 days at room temp"},
    "strawberry":  {"max_hours": 168, "note": "5-7 days refrigerated"},
    "mango":       {"max_hours": 168, "note": "5-7 days ripe at room temp"},
    "potato":      {"max_hours": 840, "note": "Up to 5 weeks cool dark place"},
    "carrot":      {"max_hours": 672, "note": "Up to 4 weeks refrigerated"},
    "cucumber":    {"max_hours": 168, "note": "About 1 week refrigerated"},
    "bellpepper":  {"max_hours": 336, "note": "Up to 2 weeks refrigerated"},
    "capsicum":    {"max_hours": 336, "note": "Up to 2 weeks refrigerated"},
    "okra":        {"max_hours": 96,  "note": "3-4 days refrigerated"},
    "bittergourd": {"max_hours": 120, "note": "4-5 days refrigerated"},
}


# ══════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════
def load_models():
    global classifier, class_names, img_size, yolo, img_transform, model_name

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found")
        return False

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    # Detect model type from checkpoint
    saved_model_name = ckpt.get('model_name', None)
    img_size = ckpt.get('img_size', 256)

    if saved_model_name and 'swin' in saved_model_name:
        # SwinV2 model
        model_name = saved_model_name
        print(f"Loading SwinV2 model: {model_name}")
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=2,
            img_size=img_size,
        )
    else:
        # Fallback to ResNet50 (backward compatible)
        model_name = "resnet50"
        img_size = 224
        print("Loading ResNet50 model (legacy)")
        from torchvision import models
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE).eval()
    classifier = model

    if 'classes' in ckpt:
        class_names = ckpt['classes']

    img_transform = build_transform(img_size)

    print(f"✅ Classifier loaded: {class_names} | Model: {model_name} | Size: {img_size}")
    if ckpt.get('val_acc'):
        print(f"   Val accuracy: {ckpt['val_acc']:.4f}")

    # Load YOLOv8
    try:
        from ultralytics import YOLO
        yolo = YOLO("yolov8n.pt")
        yolo.to(DEVICE)
        print("✅ YOLOv8 loaded")
    except Exception as e:
        print(f"⚠️ YOLOv8: {e}")

    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

    return True


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS scans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, fruit_type TEXT, freshness TEXT,
        confidence REAL, fresh_prob REAL, rotten_prob REAL,
        shelf_life_hours REAL, action TEXT,
        discount_pct INTEGER, price_tag TEXT,
        explanation TEXT, batch_id TEXT
    )""")
    conn.commit()
    conn.close()

init_db()
load_models()


# ══════════════════════════════════════════════════════════════
# CLASSIFIER
# ══════════════════════════════════════════════════════════════
def classify_freshness(image_pil):
    tensor = img_transform(image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = classifier(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    fi = class_names.index("fresh") if "fresh" in class_names else 0
    ri = class_names.index("rotten") if "rotten" in class_names else 1
    top = probs.argmax().item()

    return {
        "freshness": class_names[top],
        "confidence": float(probs[top]),
        "fresh_prob": round(float(probs[fi]), 4),
        "rotten_prob": round(float(probs[ri]), 4),
    }


# ══════════════════════════════════════════════════════════════
# ITEM DETECTION
# ══════════════════════════════════════════════════════════════
def detect_items(image_pil):
    w, h = image_pil.size
    total_area = w * h
    min_box = total_area * 0.02
    max_box = total_area * 0.85
    boxes = []

    if yolo is not None:
        try:
            results = yolo(image_pil, conf=0.20, iou=0.45, verbose=False)
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                pad = 10
                x1, y1 = max(0, int(x1)-pad), max(0, int(y1)-pad)
                x2, y2 = min(w, int(x2)+pad), min(h, int(y2)+pad)

                area = (x2-x1) * (y2-y1)
                if area < min_box or area > max_box:
                    continue
                aspect = (x2-x1) / max(y2-y1, 1)
                if aspect > 4.0 or aspect < 0.25:
                    continue
                if (x2-x1) < 40 or (y2-y1) < 40:
                    continue
                boxes.append([x1, y1, x2, y2])
        except Exception as e:
            print(f"YOLO: {e}")

    # NMS
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    keep = []
    for a in boxes:
        ok = True
        for b in keep:
            ix1, iy1 = max(a[0],b[0]), max(a[1],b[1])
            ix2, iy2 = min(a[2],b[2]), min(a[3],b[3])
            inter = max(0,ix2-ix1)*max(0,iy2-iy1)
            aa = (a[2]-a[0])*(a[3]-a[1])
            ab = (b[2]-b[0])*(b[3]-b[1])
            if inter/(aa+ab-inter+1e-6) > 0.35:
                ok = False
                break
        if ok:
            keep.append(a)

    return keep if keep else [[0, 0, w, h]]


# ══════════════════════════════════════════════════════════════
# DECISION ENGINE — Improved shelf life using texture confidence
# ══════════════════════════════════════════════════════════════
def get_decision(fruit_type, freshness, confidence, fresh_prob, rotten_prob):
    """
    Shelf life calculation using BOTH the classification AND the probability spread.
    
    Key insight: A fresh item with fresh_prob=0.98 is clearly fresh (high shelf life),
    while fresh_prob=0.55 means it's borderline (much less shelf life).
    The probability spread tells us WHERE on the freshness curve the item is.
    """
    info = SHELF_LIFE.get(fruit_type.lower(), {"max_hours": 168, "note": "~1 week"})
    max_h = info["max_hours"]

    if freshness == "rotten":
        return _build_result(fruit_type, "rotten", confidence, fresh_prob, rotten_prob,
                             0, "0 hours", max_h, info, "compost", 100, "REMOVE — COMPOST",
                             f"This {fruit_type} is rotten. Remove from shelf and compost immediately.")

    # Use the fresh_prob directly as the freshness score
    # fresh_prob close to 1.0 = very fresh, close to 0.5 = borderline
    freshness_score = fresh_prob

    # Map freshness score to remaining shelf life percentage
    # 0.95+ → 85-100% shelf life remaining (just arrived, very fresh)
    # 0.85-0.95 → 60-85% (fresh, good condition)
    # 0.70-0.85 → 35-60% (starting to age, still sellable)
    # 0.55-0.70 → 15-35% (aging, discount soon)
    # <0.55 → 5-15% (borderline, deep discount)
    if freshness_score >= 0.95:
        pct = 0.85 + (freshness_score - 0.95) * 3.0  # 0.85 to 1.0
    elif freshness_score >= 0.85:
        pct = 0.60 + (freshness_score - 0.85) * 2.5  # 0.60 to 0.85
    elif freshness_score >= 0.70:
        pct = 0.35 + (freshness_score - 0.70) * 1.67  # 0.35 to 0.60
    elif freshness_score >= 0.55:
        pct = 0.15 + (freshness_score - 0.55) * 1.33  # 0.15 to 0.35
    else:
        pct = max(0.05, freshness_score * 0.27)         # 0.05 to 0.15

    shelf = round(max_h * pct, 1)

    if shelf >= 48:
        shelf_display = f"{shelf/24:.1f} days"
    else:
        shelf_display = f"{shelf:.0f} hours"

    # Pricing thresholds (proportional to this fruit's max)
    if shelf >= max_h * 0.55:
        action, discount, tag = "full_price", 0, "FULL PRICE"
    elif shelf >= max_h * 0.25:
        action, discount, tag = "discount", 40, "40% OFF — QUICK SALE"
    else:
        action, discount, tag = "deep_discount", 65, "65% OFF — LAST CHANCE"

    reason = (f"This {fruit_type} is fresh ({fresh_prob:.0%} confidence). "
              f"Estimated shelf life: {shelf_display} out of max {max_h/24:.0f} days. "
              f"{info['note']}.")

    return _build_result(fruit_type, "fresh", confidence, fresh_prob, rotten_prob,
                         shelf, shelf_display, max_h, info, action, discount, tag, reason)


def _build_result(fruit_type, freshness, confidence, fresh_prob, rotten_prob,
                  shelf, shelf_display, max_h, info, action, discount, tag, reason):
    return {
        "fruit_type": fruit_type, "freshness": freshness,
        "confidence": round(confidence, 4),
        "fresh_prob": round(fresh_prob, 4),
        "rotten_prob": round(rotten_prob, 4),
        "shelf_life_hours": shelf,
        "shelf_life_display": shelf_display,
        "max_shelf_life_hours": max_h,
        "max_shelf_life_display": f"{max_h/24:.0f} days",
        "storage_note": info["note"],
        "action": action, "discount_percentage": discount,
        "price_tag": tag, "reason": reason,
        "timestamp": datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════
# ANNOTATE IMAGE
# ══════════════════════════════════════════════════════════════
def annotate(image_pil, items):
    draw = ImageDraw.Draw(image_pil)
    for item in items:
        bbox = item.get("bbox")
        if not bbox:
            continue

        x1, y1, x2, y2 = bbox
        is_fresh = item["freshness"] == "fresh"
        color = (0, 180, 0) if is_fresh else (220, 0, 0)

        for i in range(4):
            draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=color)

        n = item.get("item_number", "?")
        f = item["freshness"].upper()
        c = item["confidence"]
        sl = item.get("shelf_life_display", "")
        pt = item.get("price_tag", "")

        line1 = f"#{n} {f} ({c:.0%})"
        line2 = f"{sl} | {pt}" if is_fresh else "COMPOST"

        tw = max(len(line1), len(line2)) * 9 + 10
        draw.rectangle([x1, y1-40, x1+tw, y1], fill=color)
        draw.text((x1+4, y1-38), line1, fill="white")
        draw.text((x1+4, y1-20), line2, fill="white")

    buf = io.BytesIO()
    image_pil.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


# ══════════════════════════════════════════════════════════════
# LLM
# ══════════════════════════════════════════════════════════════
OLLAMA = "http://localhost:11434/api/generate"
LLM = "llama3.1:8b"

def llm(prompt, max_tok=250, timeout=45):
    try:
        r = http_requests.post(OLLAMA, json={
            "model": LLM, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.3, "num_predict": max_tok}
        }, timeout=timeout)
        if r.status_code == 200:
            return r.json().get("response", "").strip()
    except:
        pass
    return None


def explain_batch(fruit_type, items, summary):
    items_text = "\n".join(
        f"  Item #{it['item_number']}: {it['freshness']} "
        f"(fresh_prob: {it['fresh_prob']:.0%}, "
        f"shelf life: {it['shelf_life_display']}, {it['price_tag']})"
        for it in items
    )
    prompt = f"""You are FreshScan AI, an AI produce freshness scanner used by grocery store managers.
You scanned a batch of {len(items)} {fruit_type}(s). Write a professional 3-5 sentence report.
State exactly how many are fresh vs rotten. Name the item numbers that are rotten.
Give average shelf life for fresh items. Recommend specific actions.
No markdown. No bullet points. No asterisks.

Items:
{items_text}

Fresh: {summary['fresh_count']} | Rotten: {summary['rotten_count']}
Avg shelf life (fresh): {summary['avg_shelf_display']}

Respond with ONLY the report."""
    return llm(prompt, 300, 60) or summary["recommendation"]


def explain_single(result):
    prompt = f"""You are FreshScan AI. Give a 2-3 sentence explanation for a store manager. No markdown.
Produce: {result['fruit_type']} | Freshness: {result['freshness']}
Fresh probability: {result['fresh_prob']:.0%} | Rotten probability: {result['rotten_prob']:.0%}
Shelf Life: {result['shelf_life_display']} (max {result['max_shelf_life_display']})
Storage: {result['storage_note']} | Action: {result['price_tag']}
Respond with ONLY the explanation."""
    return llm(prompt, 150) or result.get("reason", "")


# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.post("/scan-batch")
async def scan_batch(file: UploadFile = File(...), fruit_type: str = Form(...)):
    if not classifier:
        return JSONResponse(status_code=503, content={"error": "Model not loaded."})

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    boxes = detect_items(image)

    items = []
    for i, bbox in enumerate(boxes):
        x1, y1, x2, y2 = bbox
        crop = image.crop((x1, y1, x2, y2))
        if crop.size[0] < 30 or crop.size[1] < 30:
            continue

        cls = classify_freshness(crop)
        dec = get_decision(fruit_type, cls["freshness"], cls["confidence"],
                           cls["fresh_prob"], cls["rotten_prob"])
        dec["item_number"] = i + 1
        dec["bbox"] = bbox
        items.append(dec)

    if not items:
        cls = classify_freshness(image)
        dec = get_decision(fruit_type, cls["freshness"], cls["confidence"],
                           cls["fresh_prob"], cls["rotten_prob"])
        dec["item_number"] = 1
        dec["bbox"] = None
        items.append(dec)

    # Summary
    fresh = [x for x in items if x["freshness"] == "fresh"]
    rotten = [x for x in items if x["freshness"] == "rotten"]
    avg_shelf = sum(x["shelf_life_hours"] for x in fresh) / len(fresh) if fresh else 0
    avg_display = f"{avg_shelf/24:.1f} days" if avg_shelf >= 48 else f"{avg_shelf:.0f} hours"

    if not rotten:
        rec = f"All {len(items)} {fruit_type}(s) are fresh. Sell at full price."
    elif not fresh:
        rec = f"All {len(items)} {fruit_type}(s) are rotten. Remove and compost."
    else:
        nums = ", #".join(str(x["item_number"]) for x in rotten)
        rec = (f"Mixed: {len(fresh)} fresh, {len(rotten)} rotten. "
               f"Remove item(s) #{nums}. Fresh avg shelf life: {avg_display}.")

    summary = {"fresh_count": len(fresh), "rotten_count": len(rotten),
               "avg_shelf_display": avg_display, "recommendation": rec}

    annotated_b64 = annotate(image.copy(), items)
    explanation = explain_batch(fruit_type, items, summary)

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    for it in items:
        _log(it, batch_id)

    return {
        "batch_id": batch_id, "fruit_type": fruit_type,
        "total_items": len(items), "fresh_count": len(fresh),
        "rotten_count": len(rotten),
        "average_shelf_life_hours": round(avg_shelf, 1),
        "average_shelf_life_display": avg_display,
        "batch_recommendation": rec, "items": items,
        "annotated_image": annotated_b64, "explanation": explanation,
    }


@app.post("/scan-single")
async def scan_single(file: UploadFile = File(...), fruit_type: str = Form(...)):
    if not classifier:
        return JSONResponse(status_code=503, content={"error": "Model not loaded."})

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    cls = classify_freshness(image)
    dec = get_decision(fruit_type, cls["freshness"], cls["confidence"],
                       cls["fresh_prob"], cls["rotten_prob"])
    dec["explanation"] = explain_single(dec)
    _log(dec, None)
    return dec


def _log(r, batch_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO scans (timestamp,fruit_type,freshness,confidence,fresh_prob,rotten_prob,"
            "shelf_life_hours,action,discount_pct,price_tag,explanation,batch_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (r.get("timestamp"), r["fruit_type"], r["freshness"], r["confidence"],
             r.get("fresh_prob", 0), r.get("rotten_prob", 0),
             r["shelf_life_hours"], r["action"], r["discount_percentage"],
             r.get("price_tag", ""), r.get("explanation", ""), batch_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB: {e}")


# ══════════════════════════════════════════════════════════════
# STATS & DAILY REPORT (FIXED)
# ══════════════════════════════════════════════════════════════

@app.get("/stats")
async def stats():
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT action, COUNT(*), AVG(confidence), AVG(shelf_life_hours) "
            "FROM scans WHERE DATE(timestamp)=DATE('now','localtime') GROUP BY action"
        ).fetchall()
        by_action = {r[0]: {"count": r[1], "avg_confidence": round(r[2], 3),
                            "avg_shelf_life": round(r[3], 1)} for r in rows}

        t = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN action='compost' THEN 1 ELSE 0 END) "
            "FROM scans WHERE DATE(timestamp)=DATE('now','localtime')"
        ).fetchone()
        total, comp = t[0] or 0, t[1] or 0
    except:
        by_action, total, comp = {}, 0, 0
    finally:
        conn.close()

    return {"by_action": by_action, "total_scans": total,
            "waste_rate": round(comp/total*100, 1) if total else 0}


@app.get("/daily-report")
async def daily_report():
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT fruit_type, freshness, action, confidence, shelf_life_hours, price_tag "
            "FROM scans WHERE DATE(timestamp)=DATE('now','localtime')"
        ).fetchall()
    except:
        rows = []
    finally:
        conn.close()

    if not rows:
        return {"report": "No scans recorded today yet. Start scanning produce to generate a report.",
                "total_scans": 0, "summary": {}}

    scans = [{"fruit_type": r[0], "freshness": r[1], "action": r[2],
              "confidence": r[3], "shelf_life_hours": r[4], "price_tag": r[5]} for r in rows]

    total = len(scans)
    fp = sum(1 for s in scans if s["action"] == "full_price")
    dc = sum(1 for s in scans if s["action"] in ("discount", "deep_discount"))
    cm = sum(1 for s in scans if s["action"] == "compost")
    avg_shelf = sum(s["shelf_life_hours"] for s in scans if s["action"] != "compost")
    fresh_count = sum(1 for s in scans if s["freshness"] == "fresh")
    avg_shelf = avg_shelf / fresh_count if fresh_count > 0 else 0

    # Get unique fruit types scanned
    fruit_types = list(set(s["fruit_type"] for s in scans))

    prompt = f"""You are FreshScan AI. Write a professional 5-6 sentence daily produce quality report
for the store manager. Include specific numbers and actionable recommendations.
No markdown. No bullet points. No asterisks.

Today's scanning data:
- Total items scanned: {total}
- Produce types: {', '.join(fruit_types)}
- Full price (fresh, good condition): {fp} items
- Discounted (aging, reduced price): {dc} items
- Composted (rotten, removed): {cm} items
- Waste rate: {cm/total*100:.1f}%
- Average shelf life of fresh items: {avg_shelf/24:.1f} days

Respond with ONLY the report."""

    report = llm(prompt, 350, 60)
    if not report:
        report = (f"Today's produce scan summary: {total} items were scanned across "
                  f"{len(fruit_types)} produce type(s) ({', '.join(fruit_types)}). "
                  f"{fp} items are at full price, {dc} items were moved to discount, "
                  f"and {cm} items were composted. "
                  f"The waste rate is {cm/total*100:.1f}%. "
                  f"Average remaining shelf life for fresh items is {avg_shelf/24:.1f} days.")

    return {
        "report": report,
        "total_scans": total,
        "summary": {
            "full_price": fp, "discounted": dc, "composted": cm,
            "waste_rate": round(cm/total*100, 1),
            "avg_shelf_life_days": round(avg_shelf/24, 1),
            "produce_types": fruit_types,
        }
    }


@app.get("/history")
async def history():
    """Get all scans from today with full details."""
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT timestamp, fruit_type, freshness, confidence, fresh_prob, rotten_prob, "
            "shelf_life_hours, action, discount_pct, price_tag, batch_id "
            "FROM scans WHERE DATE(timestamp)=DATE('now','localtime') "
            "ORDER BY timestamp DESC"
        ).fetchall()
    except:
        rows = []
    finally:
        conn.close()

    return [
        {"timestamp": r[0], "fruit_type": r[1], "freshness": r[2],
         "confidence": r[3], "fresh_prob": r[4], "rotten_prob": r[5],
         "shelf_life_hours": r[6], "action": r[7], "discount_pct": r[8],
         "price_tag": r[9], "batch_id": r[10]}
        for r in rows
    ]


@app.get("/health")
async def health():
    return {
        "status": "healthy", "version": "4.0",
        "model": model_name,
        "classifier_loaded": classifier is not None,
        "yolo_loaded": yolo is not None,
        "img_size": img_size,
        "classes": class_names,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
