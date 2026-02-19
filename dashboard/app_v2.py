"""
app_v2.py — Upgraded Streamlit Dashboard
- No dropdown needed — auto-detects fruit type
- Multi-item detection and display
- Per-item freshness with bounding boxes
- Average shelf life calculation
- Detailed LLM explanation mentioning each item

Run: streamlit run dashboard/app_v2.py --server.port 8501 --server.address 0.0.0.0
"""
import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io

API_URL = "http://localhost:8000"

st.set_page_config(page_title="FreshScan AI v2", page_icon="🍎", layout="wide")

# ============================================
# Header
# ============================================
st.title("🍎 FreshScan AI v2 — Smart Grocery Freshness Scanner")
st.caption("Auto-detects fruit type • Multi-item scanning • Powered by Dell Pro Max GB10 • 100% Local AI")

# Sidebar
with st.sidebar:
    st.header("System Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        if health.get("classifier_loaded"):
            st.success(f"✅ Classifier: {health.get('num_classes', '?')} classes")
        else:
            st.warning("⚠️ Classifier not loaded")
        if health.get("yolo_loaded"):
            st.success("✅ YOLOv8 multi-item detection")
        else:
            st.info("ℹ️ Single-item mode (no YOLO)")
        st.info(f"GPU: {health.get('gpu', 'Unknown')}")
    except Exception:
        st.error("❌ Backend not connected")

    st.divider()
    st.header("Today's Stats")
    try:
        stats = requests.get(f"{API_URL}/stats", timeout=5).json()
        st.metric("Total Scans", stats.get("total_scans", 0))
        st.metric("Waste Rate", f"{stats.get('waste_rate', 0):.1f}%")
        if stats.get("by_action"):
            for action, data in stats["by_action"].items():
                label = action.replace("_", " ").title()
                st.metric(label, f"{data['count']} items")
    except Exception:
        st.info("No stats yet")

# ============================================
# Main
# ============================================
tab1, tab2, tab3 = st.tabs(["📷 Scan Produce", "📝 Daily Report", "ℹ️ How It Works"])

with tab1:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Upload Produce Image")
        st.caption("Just upload — AI automatically detects what it is!")

        upload_method = st.radio("Input method:", ["Upload Image", "Camera"], horizontal=True)

        if upload_method == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        else:
            uploaded_file = st.camera_input("Take a photo")

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Your image", use_container_width=True)

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                multi_scan = st.button("🔍 Smart Scan (Multi-Item)", type="primary",
                                       use_container_width=True)
            with col_btn2:
                single_scan = st.button("📸 Quick Scan (Single Item)",
                                        use_container_width=True)

            if multi_scan or single_scan:
                endpoint = "/scan" if multi_scan else "/scan-single"
                with st.spinner("🤖 AI is analyzing..."):
                    try:
                        files = {"file": ("image.jpg", uploaded_file.getvalue(), "image/jpeg")}
                        response = requests.post(
                            f"{API_URL}{endpoint}",
                            files=files,
                            timeout=120
                        )
                        if response.status_code == 200:
                            st.session_state["result"] = response.json()
                        else:
                            st.error(f"Error: {response.json()}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to backend. Is the server running?")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    with col_right:
        st.subheader("Results")

        if "result" in st.session_state:
            r = st.session_state["result"]

            # Check if multi-item response
            is_multi = r.get("multi_item", False)

            if is_multi and "items" in r:
                # ========== MULTI-ITEM DISPLAY ==========
                st.markdown(f"### 📦 Detected {r['total_items']} items")

                # Summary metrics
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Fresh Items", r["fresh_count"])
                with c2:
                    st.metric("Rotten Items", r["rotten_count"])
                with c3:
                    st.metric("Avg Shelf Life", r["average_shelf_life_display"])

                # Per-item cards
                for i, item in enumerate(r["items"]):
                    icon = "🟢" if item["freshness"] == "fresh" else "🔴"
                    with st.container(border=True):
                        ic1, ic2, ic3, ic4 = st.columns([1, 1, 1, 1])
                        with ic1:
                            st.markdown(f"**{icon} {item['fruit_type'].title()}**")
                        with ic2:
                            st.caption(f"**{item['freshness'].upper()}**")
                        with ic3:
                            st.caption(f"Shelf: {item['shelf_life_display']}")
                        with ic4:
                            if item["action"] == "full_price":
                                st.caption("✅ Full Price")
                            elif item["action"] == "compost":
                                st.caption("🚫 Compost")
                            elif item["action"] == "discount":
                                st.caption("🏷️ 40% Off")
                            else:
                                st.caption("🏷️ 65% Off")

                # LLM explanation
                st.divider()
                st.caption("🤖 AI Analysis:")
                st.info(r.get("explanation", "No explanation available."))

            else:
                # ========== SINGLE ITEM DISPLAY ==========
                action_styles = {
                    "full_price": "🟢",
                    "discount": "🟡",
                    "deep_discount": "🟠",
                    "compost": "🔴",
                }
                icon = action_styles.get(r.get("action", ""), "⚪")

                with st.container(border=True):
                    st.markdown(f"### {icon} {r.get('fruit_type', 'Unknown').title()}")
                    st.caption("*Auto-detected — no manual selection needed*")

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Freshness", r.get("freshness", "?").title())
                    with c2:
                        st.metric("Shelf Life", r.get("shelf_life_display", "?"))
                    with c3:
                        st.metric("Confidence", f"{r.get('confidence', 0):.0%}")

                    # Storage info
                    if r.get("storage_note"):
                        st.caption(f"📦 Storage: {r['storage_note']}")

                    # Action banner
                    action = r.get("action", "")
                    tag = r.get("price_tag", "")
                    if action == "full_price":
                        st.success(f"**{tag}**")
                    elif action == "compost":
                        st.error(f"**{tag}**")
                    else:
                        st.warning(f"**{tag}**")

                    # Top predictions
                    if r.get("top5_predictions"):
                        st.caption("Top Predictions:")
                        for cls, prob in list(r["top5_predictions"].items())[:5]:
                            col_name, col_bar = st.columns([2, 3])
                            with col_name:
                                st.write(cls.replace("_", " ").title())
                            with col_bar:
                                st.progress(min(prob, 1.0))

                    # LLM Explanation
                    st.divider()
                    st.caption("🤖 AI Explanation:")
                    st.info(r.get("explanation", r.get("reason", "")))

        else:
            st.info("Upload an image and click 'Smart Scan' or 'Quick Scan' to see results.")

with tab2:
    st.subheader("📝 AI-Generated Daily Report")
    if st.button("Generate Report", type="primary"):
        with st.spinner("Generating report with local LLM..."):
            try:
                report = requests.get(f"{API_URL}/daily-report", timeout=120).json()
                st.write(report.get("report", "No report available."))
                st.caption(f"Based on {report.get('total_scans', 0)} scans today.")
            except Exception:
                st.error("Could not generate report.")

with tab3:
    st.subheader("How FreshScan AI Works")
    st.markdown("""
**Three AI Models Running Simultaneously on Dell Pro Max GB10:**

**1. YOLOv8 — Object Detection**
Finds all produce items in a single photo. Draws bounding boxes around each one.

**2. ResNet50 — 26-Class Classifier**
For each detected item, classifies both the fruit TYPE and FRESHNESS.
Trained on 31,000+ images across 13 fruit/vegetable types × 2 states (fresh/rotten).

**3. Llama 3.1 8B — Natural Language AI**
Generates human-readable explanations and daily reports.
Runs entirely on-device — no cloud, no data leaves the machine.

**Shelf Life Estimation:**
Based on USDA FoodKeeper guidelines, adjusted by classifier confidence.
Each fruit has its own maximum shelf life (e.g., potatoes last 5 weeks, strawberries 1 week).

**Decision Rules:**
- More than 60% shelf life remaining → Full Price
- 25-60% remaining → 40% Discount
- Less than 25% remaining → 65% Deep Discount
- Rotten → Remove & Compost
    """)
