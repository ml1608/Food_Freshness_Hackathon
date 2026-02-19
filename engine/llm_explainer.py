"""
llm_explainer.py — LLM explanations via Ollama
"""
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"


def _call_llm(prompt, max_tokens=200, timeout=30):
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": max_tokens}
        }, timeout=timeout)
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
    except Exception:
        pass
    return None


def explain_single(result):
    prompt = f"""You are FreshScan AI, a grocery produce freshness scanner.
Give a 2-3 sentence explanation for the store manager. Be specific. No markdown.

Produce: {result['fruit_type']}
Freshness: {result['freshness']}
Confidence: {result['confidence']:.0%}
Shelf Life Remaining: {result['shelf_life_display']}
Max Shelf Life: {result['max_shelf_life_display']}
Storage: {result['storage_note']}
Action: {result['price_tag']}

Respond with ONLY the explanation."""

    text = _call_llm(prompt)
    return text or result.get("reason", "Analysis complete.")


def explain_batch(batch_summary):
    items_text = ""
    for item in batch_summary["items"]:
        items_text += (f"  Item #{item['item_number']}: {item['freshness']} "
                       f"(confidence {item['confidence']:.0%}, "
                       f"shelf life {item['shelf_life_display']})\n")

    prompt = f"""You are FreshScan AI, a grocery produce freshness scanner.
You scanned a batch of {batch_summary['total_items']} {batch_summary['fruit_type']}(s).
Write a 3-5 sentence report for the store manager.
Clearly state how many are fresh and how many are rotten.
Mention which item numbers are rotten.
Include average shelf life for fresh items.
Give specific action. No markdown. No bullet points.

Items:
{items_text}
Fresh: {batch_summary['fresh_count']} | Rotten: {batch_summary['rotten_count']}
Avg shelf life (fresh): {batch_summary['average_shelf_life_display']}

Respond with ONLY the report."""

    text = _call_llm(prompt, max_tokens=300, timeout=60)
    return text or batch_summary.get("batch_recommendation", "Batch analysis complete.")


def explain_daily(scans):
    total = len(scans)
    if total == 0:
        return "No scans today."

    full = sum(1 for s in scans if s["action"] == "full_price")
    disc = sum(1 for s in scans if s["action"] in ("discount", "deep_discount"))
    comp = sum(1 for s in scans if s["action"] == "compost")

    prompt = f"""You are FreshScan AI. Write a 4-5 sentence daily report. Professional. No markdown.
Scanned: {total} | Full price: {full} | Discounted: {disc} | Composted: {comp} | Waste rate: {comp/total*100:.1f}%
Respond with ONLY the report."""

    text = _call_llm(prompt, max_tokens=250, timeout=60)
    return text or f"Daily: {total} scanned, {full} full price, {disc} discounted, {comp} composted."
