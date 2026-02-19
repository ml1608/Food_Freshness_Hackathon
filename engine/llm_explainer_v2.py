"""
llm_explainer_v2.py — Upgraded LLM explainer with multi-item support
Run: python3 engine/llm_explainer_v2.py  (to test)
"""
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"


def generate_explanation(scan_result):
    """Generate explanation for a single item."""
    prompt = f"""You are FreshScan AI, a grocery store produce freshness analysis system.
Based on this scan, give a 2-3 sentence explanation for the store manager.
Be specific and professional. Do NOT use markdown.

Fruit: {scan_result['fruit_type']}
Freshness: {scan_result['freshness']}
Confidence: {scan_result['confidence']:.0%}
Remaining Shelf Life: {scan_result['shelf_life_display']}
Maximum Shelf Life: {scan_result['max_shelf_life']/24:.0f} days
Storage: {scan_result['storage_note']}
Action: {scan_result['action']}
Discount: {scan_result['discount_percentage']}%

Respond with ONLY the explanation, nothing else."""

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 150}
        }, timeout=30)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        return scan_result.get("reason", "Explanation unavailable.")
    except Exception:
        return scan_result.get("reason", "LLM not available.")


def generate_multi_item_explanation(summary):
    """Generate explanation for multiple items detected in one image."""

    items_desc = ""
    for item in summary["items"]:
        items_desc += (f"- {item['fruit_type']}: {item['freshness']} "
                       f"(confidence {item['confidence']:.0%}, "
                       f"shelf life: {item['shelf_life_display']}, "
                       f"action: {item['action']})\n")

    prompt = f"""You are FreshScan AI, a grocery store produce freshness analysis system.
You scanned an image containing {summary['total_items']} items. Give a 4-5 sentence report.
Clearly state which items are FRESH and which are ROTTEN by name.
Include the average shelf life for fresh items.
Give specific action recommendations. Do NOT use markdown or bullet points.

Items detected:
{items_desc}
Fresh items: {summary['fresh_count']}
Rotten items: {summary['rotten_count']}
Average shelf life (fresh items): {summary['average_shelf_life_display']}

Respond with ONLY the report, nothing else."""

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 250}
        }, timeout=60)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception:
        pass

    # Fallback
    return (f"Scanned {summary['total_items']} items: "
            f"{summary['fresh_count']} fresh ({summary['fresh_items_summary']}), "
            f"{summary['rotten_count']} rotten ({summary['rotten_items_summary']}). "
            f"Average shelf life for fresh items: {summary['average_shelf_life_display']}.")


def generate_daily_report(scan_history):
    """Generate end-of-day summary report."""
    total = len(scan_history)
    if total == 0:
        return "No scans recorded today."

    full_price = sum(1 for s in scan_history if s["action"] == "full_price")
    discounted = sum(1 for s in scan_history if s["action"] in ["discount", "deep_discount"])
    composted = sum(1 for s in scan_history if s["action"] == "compost")
    waste_rate = composted / total * 100

    prompt = f"""You are FreshScan AI. Write a 4-5 sentence daily produce report for the store manager.
Be professional. Include actionable recommendations. Do NOT use markdown.

Today's data:
- Total scanned: {total}
- Full price: {full_price}
- Discounted: {discounted}
- Composted: {composted}
- Waste rate: {waste_rate:.1f}%

Respond with ONLY the report."""

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.4, "num_predict": 250}
        }, timeout=60)
        return response.json().get("response", "").strip()
    except Exception:
        return (f"Daily summary: {total} items scanned. {full_price} at full price, "
                f"{discounted} discounted, {composted} composted. "
                f"Waste rate: {waste_rate:.1f}%.")


# Test
if __name__ == "__main__":
    from decision_engine_v2 import decide_action, analyze_multi_items

    print("Test 1: Single item explanation")
    result = decide_action("carrot", "fresh", 0.92)
    explanation = generate_explanation(result)
    print(f"  {explanation}\n")

    print("Test 2: Multi-item explanation")
    items = [
        {"fruit_type": "banana", "freshness": "fresh", "confidence": 0.90},
        {"fruit_type": "apple", "freshness": "rotten", "confidence": 0.88},
    ]
    summary = analyze_multi_items(items)
    multi_explanation = generate_multi_item_explanation(summary)
    print(f"  {multi_explanation}")
