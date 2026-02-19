"""
app.py — FreshScan AI Dashboard (Final)
Run: streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
"""



import time


import streamlit as st
import requests
from PIL import Image
import io
import base64

API = "http://localhost:8000"

PRODUCE = [
    "apple", "banana", "orange", "tomato", "strawberry",
    "mango", "potato", "carrot", "cucumber", "bellpepper",
    "capsicum", "okra", "bittergourd",
]

SHELF_REF = {
    "apple": "2 weeks refrigerated", "banana": "5-7 days room temp",
    "orange": "3 weeks refrigerated", "tomato": "5-7 days room temp",
    "strawberry": "5-7 days refrigerated", "mango": "5-7 days ripe",
    "potato": "5 weeks cool dark place", "carrot": "4 weeks refrigerated",
    "cucumber": "1 week refrigerated", "bellpepper": "2 weeks refrigerated",
    "capsicum": "2 weeks refrigerated", "okra": "3-4 days refrigerated",
    "bittergourd": "4-5 days refrigerated",
}

st.set_page_config(page_title="FreshScan AI", page_icon="🍎", layout="wide")
st.title("🍎 FreshScan AI — Grocery Freshness Scanner")
st.caption("Per-item detection • Shelf life & pricing • Powered by Dell Pro Max GB10 • 100% Local AI")

st.set_page_config(page_title="FreshScan AI", page_icon="🍎", layout="wide")

if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = 0



st.warning(
    "⚠️ **Important:** Results are based only on the visible side of the produce in the uploaded image. "
    "To check the entire surface area, please turn the produce and scan again from different angles. "
    "For best results, scan each side separately."
)

# ── Sidebar ──
with st.sidebar:
    st.header("System")
    try:
        h = requests.get(f"{API}/health", timeout=5).json()
        if h.get("classifier_loaded"):
            st.success(f"✅ Model: {h.get('model', '?')}")
        if h.get("verifier_loaded"):
            st.success("✅ Fruit verifier active")
        if h.get("yolo_loaded"):
            st.success("✅ YOLOv8 active")
        st.info(f"GPU: {h.get('gpu', '?')}")
    except:
        st.error("❌ Backend offline")

    st.divider()
    try:
        s = requests.get(f"{API}/stats", timeout=5).json()
        c1, c2 = st.columns(2)
        c1.metric("Scans Today", s.get("total_scans", 0))
        c2.metric("Waste Rate", f"{s.get('waste_rate', 0):.1f}%")
    except:
        pass

    st.divider()
    st.header("Shelf Life Guide")
    for p, sl in SHELF_REF.items():
        st.caption(f"**{p.title()}:** {sl}")

# ── Tabs ──
tab1, tab2, tab3, tab4 = st.tabs(["📦 Batch Scan", "📸 Single Scan", "📝 Daily Report", "ℹ️ About"])

# ══════════════════════════════════════════
# BATCH SCAN
# ══════════════════════════════════════════
with tab1:
    st.subheader("📦 Batch Scan")
    st.markdown("Select produce → Upload photo → Get per-item freshness, shelf life, and pricing")

    left, right = st.columns([1, 1])

    with left:
        fruit = st.selectbox("What produce is this?", PRODUCE, format_func=str.title, key="bf")
        st.caption(f"Max shelf life: **{SHELF_REF.get(fruit, '?')}**")

        method = st.radio("Input:", ["Upload", "Camera"], horizontal=True, key="bm")
        up = (st.file_uploader("Upload batch photo", type=["jpg", "jpeg", "png"], key="bu")
              if method == "Upload"
              else st.camera_input("Take photo", key="bc"))

        if up:
            st.image(Image.open(up), caption=f"Your {fruit}(s)", use_container_width=True)

            if st.button("🔍 Scan Batch", type="primary", use_container_width=True):
                with st.spinner(f"Detecting and analyzing each {fruit}..."):
                    try:
                        r = requests.post(f"{API}/scan-batch",
                                          files={"file": ("img.jpg", up.getvalue(), "image/jpeg")},
                                          data={"fruit_type": fruit}, timeout=120)
                        if r.ok:
                            st.session_state["batch"] = r.json()
                        elif r.status_code == 400:
                            err = r.json()
                            if err.get("error") == "fruit_type_mismatch":
                                st.error(
                                    f"⚠️ **Wrong produce type selected!**\n\n"
                                    f"You selected **{err.get('selected_type', '?').title()}** "
                                    f"but the image looks like **{err.get('detected_type', '?').title()}** "
                                    f"({err.get('detected_confidence', 0):.0%} confidence).\n\n"
                                    f"Please select the correct produce type from the dropdown and try again."
                                )
                            else:
                                st.error(f"Error: {err.get('message', r.text)}")
                        else:
                            st.error(r.text)
                    except requests.exceptions.ConnectionError:
                        st.error("Backend not running! Start the API server first.")
                    except Exception as e:
                        st.error(str(e))

    with right:
        if "batch" in st.session_state:
            b = st.session_state["batch"]

            if b.get("annotated_image"):
                img = Image.open(io.BytesIO(base64.b64decode(b["annotated_image"])))
                st.image(img, caption="🟢 Fresh  |  🔴 Rotten", use_container_width=True)

            st.markdown(f"### {b.get('fruit_type', '').title()} — Batch Results")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total", b.get("total_items", 0))
            m2.metric("🟢 Fresh", b.get("fresh_count", 0))
            m3.metric("🔴 Rotten", b.get("rotten_count", 0))
            m4.metric("Avg Shelf Life", b.get("average_shelf_life_display", "—"))

            rec = b.get("batch_recommendation", "")
            rc = b.get("rotten_count", 0)
            fc = b.get("fresh_count", 0)
            if rc == 0:
                st.success(f"✅ {rec}")
            elif fc == 0:
                st.error(f"🚫 {rec}")
            else:
                st.warning(f"⚠️ {rec}")

            st.markdown("#### Individual Items")
            for item in b.get("items", []):
                n = item["item_number"]
                f = item["freshness"]
                c = item["confidence"]
                sl = item.get("shelf_life_display", "—")
                pt = item.get("price_tag", "—")
                fp = item.get("fresh_prob", 0)
                rp = item.get("rotten_prob", 0)

                if f == "fresh":
                    st.success(
                        f"**#{n}** 🟢 FRESH | Confidence: {c:.0%} | "
                        f"Fresh prob: {fp:.0%} | Shelf life: {sl} | {pt}"
                    )
                else:
                    st.error(
                        f"**#{n}** 🔴 ROTTEN | Confidence: {c:.0%} | "
                        f"Rotten prob: {rp:.0%} | REMOVE & COMPOST"
                    )

            if b.get("explanation"):
                st.divider()
                st.caption("🤖 AI Analysis:")
                st.info(b["explanation"])
        else:
            st.info("👆 Select produce type, upload a photo, and click Scan Batch")

# ══════════════════════════════════════════
# SINGLE SCAN
# ══════════════════════════════════════════
with tab2:
    st.subheader("📸 Single Item Scan")
    l2, r2 = st.columns([1, 1])

    with l2:
        fruit2 = st.selectbox("Produce:", PRODUCE, format_func=str.title, key="sf")
        up2 = st.file_uploader("Upload photo", type=["jpg", "jpeg", "png"], key="su")

        if up2:
            st.image(Image.open(up2), use_container_width=True)
            if st.button("🔍 Analyze", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    try:
                        r = requests.post(f"{API}/scan-single",
                                          files={"file": ("img.jpg", up2.getvalue(), "image/jpeg")},
                                          data={"fruit_type": fruit2}, timeout=60)
                        if r.ok:
                            st.session_state["single"] = r.json()
                        elif r.status_code == 400:
                            err = r.json()
                            if err.get("error") == "fruit_type_mismatch":
                                st.error(
                                    f"⚠️ **Wrong produce type selected!**\n\n"
                                    f"You selected **{err.get('selected_type', '?').title()}** "
                                    f"but the image looks like **{err.get('detected_type', '?').title()}** "
                                    f"({err.get('detected_confidence', 0):.0%} confidence).\n\n"
                                    f"Please select the correct produce type and try again."
                                )
                            else:
                                st.error(f"Error: {err.get('message', r.text)}")
                        else:
                            st.error(r.text)
                    except Exception as e:
                        st.error(str(e))

    with r2:
        if "single" in st.session_state:
            s = st.session_state["single"]
            icon = "🟢" if s.get("freshness") == "fresh" else "🔴"

            with st.container(border=True):
                st.markdown(f"### {icon} {s.get('fruit_type', '?').title()}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Freshness", s.get("freshness", "?").title())
                c2.metric("Shelf Life", s.get("shelf_life_display", "?"))
                c3.metric("Confidence", f"{s.get('confidence', 0):.0%}")

                if s.get("storage_note"):
                    st.caption(f"📦 {s['storage_note']}")

                a = s.get("action", "")
                t = s.get("price_tag", "")
                if a == "full_price":
                    st.success(f"**{t}**")
                elif a == "compost":
                    st.error(f"**{t}**")
                else:
                    st.warning(f"**{t}**")

                fp = float(s.get("fresh_prob", 0))
                rp = float(s.get("rotten_prob", 0))
                st.caption("Classification Probabilities:")
                pc1, pc2 = st.columns([1, 3])
                pc1.write("Fresh"); pc2.progress(min(fp, 1.0))
                pc3, pc4 = st.columns([1, 3])
                pc3.write("Rotten"); pc4.progress(min(rp, 1.0))

                st.divider()
                st.caption("🤖 AI Explanation:")
                st.info(s.get("explanation", s.get("reason", "")))
        else:
            st.info("Upload an image and click Analyze")

# ══════════════════════════════════════════
# DAILY REPORT (FIXED)
# ══════════════════════════════════════════
with tab3:
    st.subheader("📝 Daily Produce Report")

    # Always show current stats
    try:
        stats = requests.get(f"{API}/stats", timeout=5).json()
        if stats.get("total_scans", 0) > 0:
            st.markdown("#### Today's Overview")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Total Scans", stats.get("total_scans", 0))
            mc2.metric("Waste Rate", f"{stats.get('waste_rate', 0):.1f}%")

            by_action = stats.get("by_action", {})
            fresh_count = by_action.get("full_price", {}).get("count", 0)
            disc_count = (by_action.get("discount", {}).get("count", 0) +
                          by_action.get("deep_discount", {}).get("count", 0))
            comp_count = by_action.get("compost", {}).get("count", 0)

            mc3.metric("🟢 Full Price", fresh_count)
            mc4.metric("🔴 Composted", comp_count)

            st.divider()
        else:
            st.info("No scans recorded today. Scan some produce first!")
    except:
        st.info("Cannot load stats. Make sure the backend is running.")

    # Generate report button
    if st.button("🤖 Generate AI Report", type="primary", use_container_width=True):
        with st.spinner("Llama 3.1 is writing your daily report..."):
            try:
                rpt = requests.get(f"{API}/daily-report", timeout=120).json()

                if rpt.get("total_scans", 0) == 0:
                    st.warning("No scans today. Scan some produce first to generate a report.")
                else:
                    st.markdown("#### AI-Generated Report")
                    st.info(rpt.get("report", "No report generated."))

                    # Show summary details
                    summary = rpt.get("summary", {})
                    if summary:
                        st.markdown("#### Detailed Breakdown")
                        dc1, dc2, dc3 = st.columns(3)
                        dc1.metric("Full Price Items", summary.get("full_price", 0))
                        dc2.metric("Discounted Items", summary.get("discounted", 0))
                        dc3.metric("Composted Items", summary.get("composted", 0))

                        if summary.get("produce_types"):
                            st.caption(f"Produce types scanned: {', '.join(t.title() for t in summary['produce_types'])}")
                        if summary.get("avg_shelf_life_days"):
                            st.caption(f"Average shelf life of fresh items: {summary['avg_shelf_life_days']} days")

            except requests.exceptions.ConnectionError:
                st.error("Backend not running!")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Scan history
    st.divider()
    st.markdown("#### Scan History")
    try:
        history = requests.get(f"{API}/history", timeout=10).json()
        if history:
            for scan in history[:20]:  # Show last 20
                ts = scan.get("timestamp", "")[:19].replace("T", " ")
                ft = scan.get("fruit_type", "?").title()
                fr = scan.get("freshness", "?")
                co = scan.get("confidence", 0)
                sl = scan.get("shelf_life_hours", 0)
                pt = scan.get("price_tag", "?")

                sl_display = f"{sl/24:.1f}d" if sl >= 48 else f"{sl:.0f}h"
                icon = "🟢" if fr == "fresh" else "🔴"

                st.caption(f"{ts} | {icon} {ft} | {fr.title()} ({co:.0%}) | Shelf: {sl_display} | {pt}")
        else:
            st.caption("No scan history yet.")
    except:
        st.caption("History unavailable.")

# ══════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════
with tab4:
    st.subheader("How FreshScan AI Works")
    st.markdown("""
**Three AI Models on Dell Pro Max GB10:**

**1. YOLOv8 — Item Detection**
Finds each individual fruit/vegetable in your photo and draws bounding boxes.

**2. SwinV2-Tiny — Freshness Classifier**
Uses shifted-window transformer attention to analyze fine textures:
surface bruising, discoloration, mold, wilting, and decay patterns.
Trained on 31,000+ produce images.

**3. Llama 3.1 8B — Natural Language Reports**
Generates professional explanations and daily reports.
Runs 100% locally — no cloud, no data leaves the machine.

---

**Shelf Life** uses USDA FoodKeeper data, adjusted by the model's
confidence. Higher fresh probability = more shelf life remaining.

**Pricing:**
- Over 55% shelf life → Full Price
- 25-55% remaining → 40% Off
- Under 25% → 65% Off
- Rotten → Remove & Compost

**Why Dell Pro Max GB10?**
Running YOLOv8 + SwinV2 + Llama 3.1 8B simultaneously needs
128GB unified memory. Consumer GPUs max at 24GB.
    """)
