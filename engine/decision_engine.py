"""
decision_engine.py — Freshness → Action decision logic with USDA shelf life
"""
from datetime import datetime


SHELF_LIFE_TABLE = {
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

DEFAULT_INFO = {"max_hours": 168, "note": "~1 week (default estimate)"}


def estimate_shelf_life(fruit_type, freshness, confidence):
    info = SHELF_LIFE_TABLE.get(fruit_type.lower(), DEFAULT_INFO)
    max_h = info["max_hours"]

    if freshness == "rotten":
        return 0.0, info

    if confidence >= 0.95:
        pct = 0.80 + (confidence - 0.95) * 4.0
    elif confidence >= 0.70:
        pct = 0.50 + (confidence - 0.70) * 1.2
    else:
        pct = 0.20 + max(0, confidence - 0.40) * 1.0

    pct = max(0.05, min(1.0, pct))
    return round(max_h * pct, 1), info


def decide_action(fruit_type, freshness, confidence):
    shelf_life, info = estimate_shelf_life(fruit_type, freshness, confidence)
    max_h = info["max_hours"]

    if shelf_life >= 48:
        shelf_display = f"{shelf_life/24:.1f} days"
    else:
        shelf_display = f"{shelf_life:.0f} hours"

    if freshness == "rotten" or shelf_life == 0:
        return {
            "fruit_type": fruit_type,
            "freshness": "rotten",
            "confidence": round(confidence, 3),
            "shelf_life_hours": 0,
            "shelf_life_display": "0 hours",
            "max_shelf_life_hours": max_h,
            "max_shelf_life_display": f"{max_h/24:.0f} days",
            "storage_note": info["note"],
            "action": "compost",
            "discount_percentage": 100,
            "price_tag": "REMOVE — COMPOST",
            "reason": f"This {fruit_type} is spoiled. Remove and compost.",
            "timestamp": datetime.now().isoformat(),
        }

    full_thresh = max_h * 0.60
    disc_thresh = max_h * 0.25

    if shelf_life >= full_thresh:
        action, discount, tag = "full_price", 0, "FULL PRICE"
    elif shelf_life >= disc_thresh:
        action, discount, tag = "discount", 40, "40% OFF — QUICK SALE"
    else:
        action, discount, tag = "deep_discount", 65, "65% OFF — LAST CHANCE"

    reason = (f"This {fruit_type} is {freshness} with ~{shelf_display} remaining "
              f"out of max {max_h/24:.0f} days. {info['note']}.")

    return {
        "fruit_type": fruit_type,
        "freshness": freshness,
        "confidence": round(confidence, 3),
        "shelf_life_hours": shelf_life,
        "shelf_life_display": shelf_display,
        "max_shelf_life_hours": max_h,
        "max_shelf_life_display": f"{max_h/24:.0f} days",
        "storage_note": info["note"],
        "action": action,
        "discount_percentage": discount,
        "price_tag": tag,
        "reason": reason,
        "timestamp": datetime.now().isoformat(),
    }


def analyze_batch(fruit_type, items):
    """Analyze a batch of items of the same type."""
    results = []
    for i, item in enumerate(items):
        decision = decide_action(fruit_type, item["freshness"], item["confidence"])
        decision["item_number"] = i + 1
        decision["fresh_prob"] = item.get("fresh_prob", 0)
        decision["rotten_prob"] = item.get("rotten_prob", 0)
        decision["bbox"] = item.get("bbox")
        results.append(decision)

    fresh = [r for r in results if r["freshness"] == "fresh"]
    rotten = [r for r in results if r["freshness"] == "rotten"]

    avg_shelf = 0
    if fresh:
        avg_shelf = sum(r["shelf_life_hours"] for r in fresh) / len(fresh)

    if avg_shelf >= 48:
        avg_display = f"{avg_shelf/24:.1f} days"
    else:
        avg_display = f"{avg_shelf:.0f} hours"

    # Build recommendation
    if not rotten:
        rec = f"All {len(results)} {fruit_type}(s) are fresh. Sell at full price."
    elif not fresh:
        rec = f"All {len(results)} {fruit_type}(s) are rotten. Remove and compost."
    else:
        rotten_nums = ", #".join(str(r["item_number"]) for r in rotten)
        rec = (f"Mixed batch: {len(fresh)} fresh, {len(rotten)} rotten. "
               f"Remove rotten item(s) #{rotten_nums}. "
               f"Fresh items avg shelf life: {avg_display}.")

    return {
        "fruit_type": fruit_type,
        "total_items": len(results),
        "fresh_count": len(fresh),
        "rotten_count": len(rotten),
        "average_shelf_life_hours": round(avg_shelf, 1),
        "average_shelf_life_display": avg_display,
        "batch_recommendation": rec,
        "items": results,
    }


if __name__ == "__main__":
    print("Testing decision engine...")
    for fruit, fresh, conf in [("potato", "fresh", 0.95), ("banana", "fresh", 0.80),
                                ("strawberry", "fresh", 0.72), ("okra", "rotten", 0.91)]:
        r = decide_action(fruit, fresh, conf)
        print(f"  {fruit}: {r['freshness']} | shelf: {r['shelf_life_display']} | {r['price_tag']}")
