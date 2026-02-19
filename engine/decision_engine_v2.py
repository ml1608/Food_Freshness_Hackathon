"""
decision_engine_v2.py — Upgraded decision logic with accurate shelf life data
Supports multi-item analysis from a single image

Run: python3 engine/decision_engine_v2.py  (to test)
"""
from datetime import datetime


# ============================================
# USDA-based shelf life in hours (at room temp for most, refrigerated noted)
# Sources: USDA FoodKeeper App, FDA guidelines
# These are MAXIMUM expected shelf life when purchased fresh
# ============================================
SHELF_LIFE_TABLE = {
    "apple":       {"max_hours": 336, "unit": "refrigerated", "note": "2 weeks refrigerated"},
    "banana":      {"max_hours": 168, "unit": "room_temp", "note": "5-7 days at room temp"},
    "orange":      {"max_hours": 504, "unit": "refrigerated", "note": "3 weeks refrigerated"},
    "tomato":      {"max_hours": 168, "unit": "room_temp", "note": "5-7 days at room temp"},
    "strawberry":  {"max_hours": 168, "unit": "refrigerated", "note": "5-7 days refrigerated"},
    "mango":       {"max_hours": 168, "unit": "room_temp", "note": "5-7 days (ripe, room temp)"},
    "potato":      {"max_hours": 840, "unit": "cool_dark", "note": "5 weeks in cool dark place"},
    "carrot":      {"max_hours": 672, "unit": "refrigerated", "note": "4 weeks refrigerated"},
    "cucumber":    {"max_hours": 168, "unit": "refrigerated", "note": "1 week refrigerated"},
    "bellpepper":  {"max_hours": 336, "unit": "refrigerated", "note": "2 weeks refrigerated"},
    "capsicum":    {"max_hours": 336, "unit": "refrigerated", "note": "2 weeks refrigerated"},
    "okra":        {"max_hours": 96,  "unit": "refrigerated", "note": "3-4 days refrigerated"},
    "bittergourd": {"max_hours": 120, "unit": "refrigerated", "note": "4-5 days refrigerated"},
}

DEFAULT_SHELF_LIFE = {"max_hours": 168, "unit": "room_temp", "note": "~1 week (default estimate)"}


def estimate_shelf_life(fruit_type, freshness, confidence):
    """
    Estimate remaining shelf life in hours.

    Logic:
    - Rotten → 0 hours
    - Fresh at 95%+ confidence → 80-100% of max shelf life remaining
    - Fresh at 70-95% confidence → 50-80% of max (showing early aging)
    - Fresh at <70% confidence → 20-50% of max (borderline)

    The confidence score acts as a proxy for "how far along the freshness spectrum"
    """
    info = SHELF_LIFE_TABLE.get(fruit_type.lower(), DEFAULT_SHELF_LIFE)
    max_hours = info["max_hours"]

    if freshness == "rotten":
        return 0.0, info

    # Map confidence to remaining life percentage
    if confidence >= 0.95:
        remaining_pct = 0.80 + (confidence - 0.95) * 4.0  # 0.80 to 1.0
    elif confidence >= 0.70:
        remaining_pct = 0.50 + (confidence - 0.70) * 1.2   # 0.50 to 0.80
    else:
        remaining_pct = 0.20 + (confidence - 0.40) * 1.0    # 0.20 to 0.50

    remaining_pct = max(0.05, min(1.0, remaining_pct))
    return round(max_hours * remaining_pct, 1), info


def decide_action(fruit_type, freshness, confidence):
    """
    Core decision logic.

    Discount thresholds adjusted to be proportional to each fruit's shelf life:
    - >60% of max shelf life remaining → Full price
    - 25-60% remaining → Discount (40% off)
    - <25% remaining → Deep discount (65% off)
    - Rotten → Compost
    """
    shelf_life, info = estimate_shelf_life(fruit_type, freshness, confidence)
    max_hours = info["max_hours"]

    if freshness == "rotten" or shelf_life == 0:
        return {
            "fruit_type": fruit_type,
            "freshness": "rotten",
            "confidence": round(confidence, 3),
            "shelf_life_hours": 0,
            "shelf_life_display": "0 hours",
            "max_shelf_life": max_hours,
            "storage_note": info["note"],
            "action": "compost",
            "discount_percentage": 100,
            "price_tag": "REMOVE — COMPOST",
            "reason": f"This {fruit_type} is spoiled. Remove from shelves and route to compost.",
            "timestamp": datetime.now().isoformat(),
        }

    # Calculate thresholds based on this fruit's max life
    full_price_threshold = max_hours * 0.60
    discount_threshold = max_hours * 0.25

    if shelf_life >= full_price_threshold:
        action, discount, tag = "full_price", 0, "FULL PRICE"
        reason = (f"This {fruit_type} is fresh with ~{shelf_life:.0f} hours "
                  f"({shelf_life/24:.1f} days) of shelf life remaining. "
                  f"Sell at full price. {info['note']}.")
    elif shelf_life >= discount_threshold:
        action, discount, tag = "discount", 40, "40% OFF — QUICK SALE"
        reason = (f"This {fruit_type} has ~{shelf_life:.0f} hours "
                  f"({shelf_life/24:.1f} days) remaining out of a max ~{max_hours/24:.0f} days. "
                  f"Move to discount rack for quick sale.")
    else:
        action, discount, tag = "deep_discount", 65, "65% OFF — LAST CHANCE"
        reason = (f"This {fruit_type} has only ~{shelf_life:.0f} hours "
                  f"({shelf_life/24:.1f} days) left. "
                  f"Deep discount or donate to food bank immediately.")

    # Friendly shelf life display
    if shelf_life >= 48:
        display = f"{shelf_life/24:.1f} days"
    else:
        display = f"{shelf_life:.0f} hours"

    return {
        "fruit_type": fruit_type,
        "freshness": freshness,
        "confidence": round(confidence, 3),
        "shelf_life_hours": shelf_life,
        "shelf_life_display": display,
        "max_shelf_life": max_hours,
        "storage_note": info["note"],
        "action": action,
        "discount_percentage": discount,
        "price_tag": tag,
        "reason": reason,
        "timestamp": datetime.now().isoformat(),
    }


def analyze_multi_items(items):
    """
    Analyze multiple detected items from a single image.
    Returns per-item decisions + aggregate summary.
    """
    results = []
    for item in items:
        result = decide_action(item["fruit_type"], item["freshness"], item["confidence"])
        results.append(result)

    # Aggregate stats
    fresh_items = [r for r in results if r["freshness"] == "fresh"]
    rotten_items = [r for r in results if r["freshness"] == "rotten"]

    avg_shelf_life = 0
    if fresh_items:
        avg_shelf_life = sum(r["shelf_life_hours"] for r in fresh_items) / len(fresh_items)

    summary = {
        "total_items": len(results),
        "fresh_count": len(fresh_items),
        "rotten_count": len(rotten_items),
        "average_shelf_life_hours": round(avg_shelf_life, 1),
        "average_shelf_life_display": f"{avg_shelf_life/24:.1f} days" if avg_shelf_life >= 48 else f"{avg_shelf_life:.0f} hours",
        "items": results,
        "fresh_items_summary": ", ".join(
            f"{r['fruit_type']} ({r['shelf_life_display']})" for r in fresh_items
        ) if fresh_items else "None",
        "rotten_items_summary": ", ".join(
            f"{r['fruit_type']}" for r in rotten_items
        ) if rotten_items else "None",
        "overall_action": "compost" if len(rotten_items) > len(fresh_items) else "mixed" if rotten_items else "full_price",
    }

    return summary


# ============================================
# Test
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  DECISION ENGINE v2 — TESTS")
    print("=" * 60)

    # Single item tests
    tests = [
        ("potato", "fresh", 0.97),    # Should show high shelf life (~800h)
        ("carrot", "fresh", 0.92),     # Should show ~500h+
        ("banana", "fresh", 0.85),     # Should show ~100h
        ("strawberry", "fresh", 0.72), # Should show borderline
        ("apple", "rotten", 0.91),
        ("okra", "fresh", 0.60),       # Short shelf life + low confidence
    ]

    print("\n  SINGLE ITEM ANALYSIS:")
    print("-" * 60)
    for fruit, freshness, conf in tests:
        result = decide_action(fruit, freshness, conf)
        print(f"\n  {fruit.upper()} | {freshness} | confidence: {conf:.0%}")
        print(f"    Shelf life: {result['shelf_life_display']} (max: {result['max_shelf_life']/24:.0f} days)")
        print(f"    Action:     {result['action']} | {result['price_tag']}")
        print(f"    Storage:    {result['storage_note']}")

    # Multi-item test (simulating image with 4 items)
    print(f"\n{'=' * 60}")
    print("  MULTI-ITEM ANALYSIS (simulating 4 items in one photo):")
    print("-" * 60)

    multi_items = [
        {"fruit_type": "banana", "freshness": "fresh", "confidence": 0.92},
        {"fruit_type": "apple", "freshness": "fresh", "confidence": 0.88},
        {"fruit_type": "tomato", "freshness": "rotten", "confidence": 0.95},
        {"fruit_type": "orange", "freshness": "rotten", "confidence": 0.87},
    ]

    summary = analyze_multi_items(multi_items)
    print(f"\n  Total items: {summary['total_items']}")
    print(f"  Fresh: {summary['fresh_count']} | Rotten: {summary['rotten_count']}")
    print(f"  Average shelf life (fresh only): {summary['average_shelf_life_display']}")
    print(f"  Fresh items: {summary['fresh_items_summary']}")
    print(f"  Rotten items: {summary['rotten_items_summary']}")
    print("=" * 60)
