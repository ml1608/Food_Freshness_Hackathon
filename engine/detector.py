"""
detector.py — Multi-item produce detector (Fixed)
Stricter filtering to avoid false detections from shadows/background.
"""
import cv2
import numpy as np
from PIL import Image


def find_items(image_pil, min_size_pct=0.03):
    """
    Find individual produce items in an image.
    
    Args:
        image_pil: PIL Image
        min_size_pct: Minimum item size as percentage of image area (default 3%)
                      Increase this if getting too many false detections.
    
    Returns list of bounding boxes [x1, y1, x2, y2].
    """
    img_np = np.array(image_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    total_area = h * w

    min_area = total_area * min_size_pct
    max_area = total_area * 0.80

    # ── Step 1: Color-based segmentation ──
    # Convert to HSV — find objects that are colorful (not gray/white/black background)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Keep only pixels with decent saturation AND value (eliminates shadows, white bg, dark bg)
    mask = cv2.inRange(hsv, np.array([0, 40, 50]), np.array([180, 255, 255]))

    # ── Step 2: Heavy cleanup to merge fragments and remove noise ──
    # Large kernel to close gaps within individual items
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # Close: fill holes within items
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=4)
    # Open: remove small noise spots
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=3)
    # One more close to merge any remaining fragments
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)

    # ── Step 3: Find contours ──
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # Filter extreme shapes (too thin/wide = probably not a fruit)
        aspect = bw / max(bh, 1)
        if aspect > 4.0 or aspect < 0.25:
            continue

        # Filter very small absolute size
        if bw < 50 or bh < 50:
            continue

        # Add padding
        pad_x = int(bw * 0.08)
        pad_y = int(bh * 0.08)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        boxes.append([x1, y1, x2, y2])

    # ── Step 4: Aggressive NMS to remove overlaps ──
    boxes = _nms(boxes, iou_threshold=0.30)

    # ── Step 5: Remove boxes that contain other boxes (parent/child) ──
    boxes = _remove_parent_boxes(boxes)

    if not boxes:
        # Fallback: whole image is one item
        return [[0, 0, w, h]]

    return boxes


def _nms(boxes, iou_threshold=0.30):
    """Non-maximum suppression — keep only non-overlapping boxes."""
    if len(boxes) <= 1:
        return boxes

    # Sort by area descending
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    keep = []

    for box_a in boxes:
        should_keep = True
        for box_b in keep:
            if _iou(box_a, box_b) > iou_threshold:
                should_keep = False
                break
            # Also check if one box is mostly inside the other
            if _contained(box_a, box_b) > 0.70 or _contained(box_b, box_a) > 0.70:
                should_keep = False
                break
        if should_keep:
            keep.append(box_a)

    return keep


def _remove_parent_boxes(boxes):
    """If a large box fully contains a smaller box, remove the large one."""
    if len(boxes) <= 1:
        return boxes
    
    to_remove = set()
    for i, a in enumerate(boxes):
        for j, b in enumerate(boxes):
            if i == j:
                continue
            # If box b is mostly inside box a, and a is bigger, remove a
            if _contained(b, a) > 0.80:
                area_a = (a[2]-a[0]) * (a[3]-a[1])
                area_b = (b[2]-b[0]) * (b[3]-b[1])
                if area_a > area_b * 1.5:
                    to_remove.add(i)
    
    return [b for i, b in enumerate(boxes) if i not in to_remove]


def _iou(a, b):
    """Intersection over union."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def _contained(inner, outer):
    """What fraction of inner is inside outer."""
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter / inner_area if inner_area > 0 else 0
