"""
app_v3.py — Final FreshScan AI Dashboard

Features:
- Dropdown to select fruit/vegetable type
- Upload batch photo of multiple items
- Shows annotated image with bounding boxes (green=fresh, red=rotten)
- Per-item results table
- Batch summary with average shelf life
- LLM-generated explanation
- Daily report generation

Run: streamlit run dashboard/app_v3.py --server.port 8501 --server.address 0.0.0.0
"""
import streamlit as st
import requests
from PIL import Image
import io
import base64

API_URL = "http://localhost:8000"

PRODUCE_TYPES = [
    "apple", "banana", "orange", "tomato", "strawberry",
    "mango", "potato", "carrot", "cucumber", "bellpepper",
    "capsicum", "okra", "bittergourd",
]

# Shelf life reference for display
SHELF_LIFE_REF = {
    "apple": "Up to 2 weeks (refrigerated)",
    "banana": "5-7 days (room temp)",
    "orange": "Up to 3 weeks (refrigerated)",
    "tomato": "5-7 days (room temp)",
    "strawberry": "5-7 days (refrigerated)",
    "mango": "5-7 days (ripe, room temp)",
    "potato": "Up to 5 weeks (cool dark place)",
    "carrot": "Up to 4 weeks (refrigerated)",
    "cucumber": "About 1 week (refrigerated)",
    "bellpepper": "Up to 2 weeks (refrigerated)",
    "capsicum": "Up to 2 weeks (refrigerated)",
    "okra": "3-4 days (refrigerated)",
    "bittergourd": "4-5 days (refrigerated)",
}

st.set_page_config(page_title="FreshScan AI", page_icon="🍎", layout="wide")

# ============================================
# Header
# ============================================
st.title("🍎 FreshScan AI — Grocery Freshness Scanner")
st.caption("Batch scanning • Per-item freshness detection • Powered by Dell Pro Max GB10 • 100% Local AI")

# ============================================
# Sidebar
# ============================================
with st.sidebar:
    st.header("System Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        if health.get("classifier_loaded"):
            st.success(f"✅ Model: {health.get('model_type', 'Loaded')}")
            st.info(f"GPU: {health.get('gpu', 'Unknown')}")
        else:
            st.error("❌ Model not loaded")
    except Exception:
        st.error("❌ Backend not connected. Start the API server.")

    st.divider()
    st.header("Today's Stats")
    try:
        stats = requests.get(f"{API_URL}/stats", timeout=5).json()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Scans", stats.get("total_scans", 0))
        with col2:
            st.metric("Waste Rate", f"{stats.get('waste_rate', 0):.1f}%")

        if stats.get("by_action"):
            for action, data in stats["by_action"].items():
                label = action.replace("_", " ").title()
                st.metric(label, f"{data['count']} items")
    except Exception:
        st.info("No stats yet")

    st.divider()
    st.header("Shelf Life Reference")
    for produce, shelf in SHELF_LIFE_REF.items():
        st.caption(f"**{produce.title()}:** {shelf}")

# ============================================
# Main Tabs
# ============================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📦 Batch Scan", "📸 Single Scan", "📝 Daily Report", "ℹ️ How It Works"
])

# ============================================
# TAB 1: Batch Scan
# ============================================
with tab1:
    st.subheader("📦 Batch Scan — Scan Multiple Items at Once")
    st.caption("Select the produce type, upload a photo of multiple items, and get per-item results.")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        # Fruit type selector
        fruit_type = st.selectbox(
            "What type of produce is this batch?",
            PRODUCE_TYPES,
            format_func=lambda x: x.title(),
            key="batch_fruit"
        )

        st.caption(f"📦 Max shelf life: {SHELF_LIFE_REF.get(fruit_type, 'Unknown')}")

        upload_method = st.radio("Input:", ["Upload Image", "Camera"], horizontal=True, key="batch_input")

        if upload_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Upload photo of produce batch",
                type=["jpg", "jpeg", "png"],
                key="batch_upload"
            )
        else:
            uploaded_file = st.camera_input("Take a photo", key="batch_camera")

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Your batch of {fruit_type}(s)", use_container_width=True)

            if st.button("🔍 Scan Batch", type="primary", use_container_width=True, key="batch_btn"):
                with st.spinner(f"🤖 Detecting and analyzing each {fruit_type}..."):
                    try:
                        files = {"file": ("image.jpg", uploaded_file.getvalue(), "image/jpeg")}
                        data = {"fruit_type": fruit_type}
                        response = requests.post(
                            f"{API_URL}/scan-batch",
                            files=files,
                            data=data,
                            timeout=120
                        )
                        if response.status_code == 200:
                            st.session_state["batch_result"] = response.json()
                        else:
                            st.error(f"Error: {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to backend. Is the server running?")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    with col_right:
        if "batch_result" in st.session_state:
            r = st.session_state["batch_result"]

            # Show annotated image
            if r.get("annotated_image"):
                img_bytes = base64.b64decode(r["annotated_image"])
                annotated_img = Image.open(io.BytesIO(img_bytes))
                st.image(annotated_img, caption="Detected items (🟢 Fresh, 🔴 Rotten)",
                         use_container_width=True)

            # Summary metrics
            st.markdown(f"### 📊 Batch Summary: {r.get('fruit_type', '').title()}")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Items", r.get("total_items", 0))
            with c2:
                st.metric("🟢 Fresh", r.get("fresh_count", 0))
            with c3:
                st.metric("🔴 Rotten", r.get("rotten_count", 0))
            with c4:
                st.metric("Avg Shelf Life", r.get("average_shelf_life_display", "N/A"))

            # Batch recommendation
            rec = r.get("batch_recommendation", "")
            if r.get("rotten_count", 0) == 0:
                st.success(f"✅ {rec}")
            elif r.get("fresh_count", 0) == 0:
                st.error(f"🚫 {rec}")
            else:
                st.warning(f"⚠️ {rec}")

            # Per-item results
            st.markdown("#### Per-Item Results")
            items = r.get("items", [])
            for item in items:
                num = item.get("item_number", "?")
                freshness = item.get("freshness", "?")
                conf = item.get("confidence", 0)
                shelf = item.get("shelf_life_display", "?")
                action = item.get("action", "?")

                if freshness == "fresh":
                    icon = "🟢"
                    st.success(
                        f"**Item #{num}** {icon} {freshness.upper()} — "
                        f"Confidence: {conf:.0%} — "
                        f"Shelf life: {shelf} — "
                        f"Action: {item.get('price_tag', action)}"
                    )
                else:
                    icon = "🔴"
                    st.error(
                        f"**Item #{num}** {icon} {freshness.upper()} — "
                        f"Confidence: {conf:.0%} — "
                        f"Action: REMOVE & COMPOST"
                    )

            # LLM Explanation
            if r.get("explanation"):
                st.divider()
                st.caption("🤖 AI Analysis:")
                st.info(r["explanation"])

        else:
            st.info("Select a produce type, upload a batch photo, and click 'Scan Batch' to see results.")


# ============================================
# TAB 2: Single Scan
# ============================================
with tab2:
    st.subheader("📸 Single Item Scan")

    col_left2, col_right2 = st.columns([1, 1])

    with col_left2:
        fruit_type_single = st.selectbox(
            "Produce type:",
            PRODUCE_TYPES,
            format_func=lambda x: x.title(),
            key="single_fruit"
        )

        upload_single = st.file_uploader(
            "Upload single item photo",
            type=["jpg", "jpeg", "png"],
            key="single_upload"
        )

        if upload_single:
            image_single = Image.open(upload_single)
            st.image(image_single, caption="Your image", use_container_width=True)

            if st.button("🔍 Analyze", type="primary", use_container_width=True, key="single_btn"):
                with st.spinner("Analyzing..."):
                    try:
                        files = {"file": ("image.jpg", upload_single.getvalue(), "image/jpeg")}
                        data = {"fruit_type": fruit_type_single}
                        response = requests.post(
                            f"{API_URL}/scan-single",
                            files=files,
                            data=data,
                            timeout=60
                        )
                        if response.status_code == 200:
                            st.session_state["single_result"] = response.json()
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    with col_right2:
        if "single_result" in st.session_state:
            r = st.session_state["single_result"]

            icon = "🟢" if r.get("freshness") == "fresh" else "🔴"

            with st.container(border=True):
                st.markdown(f"### {icon} {r.get('fruit_type', '?').title()}")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Freshness", r.get("freshness", "?").title())
                with c2:
                    st.metric("Shelf Life", r.get("shelf_life_display", "?"))
                with c3:
                    st.metric("Confidence", f"{r.get('confidence', 0):.0%}")

                if r.get("storage_note"):
                    st.caption(f"📦 Storage: {r['storage_note']}")

                action = r.get("action", "")
                tag = r.get("price_tag", "")
                if action == "full_price":
                    st.success(f"**{tag}**")
                elif action == "compost":
                    st.error(f"**{tag}**")
                else:
                    st.warning(f"**{tag}**")

                # Probability bars
                st.caption("Classification Probabilities:")
                fresh_p = r.get("fresh_prob", r.get("top5_predictions", {}).get("fresh", 0))
                rotten_p = r.get("rotten_prob", r.get("top5_predictions", {}).get("rotten", 0))

                fc1, fc2 = st.columns([1, 3])
                with fc1:
                    st.write("Fresh")
                with fc2:
                    st.progress(min(float(fresh_p), 1.0))

                rc1, rc2 = st.columns([1, 3])
                with rc1:
                    st.write("Rotten")
                with rc2:
                    st.progress(min(float(rotten_p), 1.0))

                st.divider()
                st.caption("🤖 AI Explanation:")
                st.info(r.get("explanation", r.get("reason", "")))
        else:
            st.info("Upload an image and click 'Analyze'.")


# ============================================
# TAB 3: Daily Report
# ============================================
with tab3:
    st.subheader("📝 AI-Generated Daily Report")
    if st.button("Generate Report", type="primary", key="report_btn"):
        with st.spinner("Generating report with local LLM..."):
            try:
                report = requests.get(f"{API_URL}/daily-report", timeout=120).json()
                st.write(report.get("report", "No report available."))
                st.caption(f"Based on {report.get('total_scans', 0)} scans today.")
            except Exception:
                st.error("Could not generate report.")


# ============================================
# TAB 4: How It Works
# ============================================
with tab4:
    st.subheader("How FreshScan AI Works")
    st.markdown("""
**Three AI Systems Running Simultaneously on Dell Pro Max GB10:**

**1. OpenCV — Multi-Item Detection**
Finds individual produce items in batch photos using computer vision
(contour detection, color segmentation, edge detection).
No need for YOLO training — works with any fruit or vegetable.

**2. ResNet50 — Freshness Classifier (98% Accuracy)**
Trained on 31,000+ images to classify produce as Fresh or Rotten.
The vendor selects the produce type, so the model focuses entirely
on the one question that matters: is it fresh or rotten?

**3. Llama 3.1 8B — Natural Language AI**
Generates human-readable batch reports and explanations.
Runs entirely on-device — no cloud, no data leaves the machine.

**Shelf Life Estimation (USDA-Based):**
Each fruit type has its own maximum shelf life from USDA FoodKeeper data.
The classifier confidence adjusts the estimate — lower confidence means
the item is likely further along the freshness curve.

**Decision Rules (Proportional to Each Fruit):**
- More than 60% of max shelf life remaining → Full Price
- 25-60% remaining → 40% Discount (Quick Sale)
- Less than 25% remaining → 65% Deep Discount (Last Chance)
- Rotten → Remove & Compost

**Why GB10?**
Running ResNet50 + OpenCV + Llama 3.1 8B simultaneously requires
significant compute and memory. The GB10's 128GB unified memory
enables all three to run together with room for the 70B model upgrade.
    """)
