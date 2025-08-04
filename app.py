# app.py
import os, io, json, time, threading, hashlib, requests, tempfile
from datetime import datetime
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF
from ultralytics import YOLO
from streamlit_autorefresh import st_autorefresh
from sklearn.linear_model import LinearRegression
import numpy as np

# ─── CONFIG ─────────────────────────────────────────────────────────────
GROQ_KEY     = os.environ.get("GROQ_API_KEY", "")  # ✅ Lee directo de Docker ENV
CAMERA_IDX   = 1
MODEL_PATH   = "yolov8n.pt"
LOGO_PATH    = "logo_stocksense.png"
COUNT_FILE   = "counts.json"
HIST_FILE    = "inventory_history.csv"
FRAME_FILE   = "last_frame.jpg"
TS_FILE      = "timestamp.txt"


# ─── PAGE SETUP ────────────────────────────────────────────────────────
st.set_page_config("StockSense AI Dashboard", layout="wide")
if not GROQ_KEY:
    st.error("⚠️ Please set GROQ_API_KEY in your `.env`.")
st_autorefresh(5000, key="auto")

# ─── SIDEBAR ────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
store_name          = st.sidebar.text_input("Store name", "La Huerta Natural")
business_goals      = st.sidebar.text_area("Business Goals / Notes", "Maximize turnover, minimize waste.")
demand_scenario     = st.sidebar.text_input("Expected Demand Scenario", "High weekend traffic")
min_threshold       = st.sidebar.slider("Low-stock threshold", 1, 20, 3)
lead_time           = st.sidebar.number_input("Lead time (days)", 1, 30, 7)
reorder_frequency   = st.sidebar.selectbox("Reorder frequency", ["Weekly", "Bi-weekly", "Monthly"])
st.sidebar.markdown("---")

# ─── SESSION STATE ───────────────────────────────────────────────────────
st.session_state.setdefault("counts_hash", None)
st.session_state.setdefault("llm_msg", "")

# ─── DETECTION THREAD ────────────────────────────────────────────────────
def draw_boxes(image, results, model):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            text = f"{label} {conf:.2f}"
            bbox = font.getbbox(text)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill="green")
            draw.text((x1, y1 - text_h), text, fill="white", font=font)
    return image

def detection_loop():
    import cv2
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_IDX)
    if not cap.isOpened():
        st.error(f"Cannot open camera {CAMERA_IDX}")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1); continue
        results = model(frame, verbose=False)
        labels = [model.names[int(b.cls[0])] for r in results for b in r.boxes]
        counts = dict(Counter(labels))
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        image = Image.fromarray(frame[:, :, ::-1])
        image = draw_boxes(image, results, model)
        image.save(FRAME_FILE)
        open(TS_FILE,"w").write(ts)
        open(COUNT_FILE,"w").write(json.dumps(counts))
        row = counts.copy(); row["timestamp"] = ts
        pd.DataFrame([row]).to_csv(
            HIST_FILE, mode="a", header=not os.path.exists(HIST_FILE), index=False
        )
        time.sleep(5)

if "det_thread" not in st.session_state:
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start(); st.session_state.det_thread = t

# ─── HELPERS ─────────────────────────────────────────────────────────────
def call_llm(counts, hist_df):
    prompt = f"""
You are an expert supply-chain consultant for “{store_name}”.
Business goals: {business_goals}
Expected demand scenario: {demand_scenario}
Current counts: {counts}
Last 7 days history: {hist_df.tail(7).to_dict(orient='records')}
Rules: reorder {reorder_frequency}, lead time {lead_time} days,
low-stock threshold {min_threshold}, no overstock.
1) Which items to reorder today?
2) Adjust reorder frequency or min stock?
3) If overstock, suggest display strategy.
Be concise, strategic, professional.
"""
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization":f"Bearer {GROQ_KEY}", "Content-Type":"application/json"},
        json={"model":"llama3-8b-8192", "messages":[{"role":"user","content":prompt}], "temperature":0.7}
    )
    if r.status_code!=200:
        return f"❌ LLM error {r.status_code}: {r.text}"
    return r.json().get("choices",[{}])[0].get("message",{}).get("content","")

def create_temp_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG")
    buf.seek(0)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.write(buf.read()); tmp.close()
    return tmp.name

def create_pdf(rtype, counts, llm_msg, hist_df):
    pdf=FPDF(); pdf.add_page()
    pdf.set_font("Helvetica","B",16); pdf.cell(0,10,f"{rtype.title()} Report",ln=1,align="C")
    if os.path.exists(LOGO_PATH):
        logo = Image.open(LOGO_PATH).convert("RGBA")
        tmp_logo = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        logo.save(tmp_logo.name); tmp_logo.close()
        pdf.image(tmp_logo.name, x=80, w=40)
    pdf.ln(5)
    pdf.set_font("Helvetica",size=12); pdf.cell(0,8,f"Generated: {datetime.now()}",ln=1); pdf.ln(5)
    pdf.set_font("Helvetica","B",14); pdf.cell(0,8,"Current Inventory:",ln=1)
    pdf.set_font("Helvetica",size=12)
    for k,v in counts.items(): pdf.cell(0,6,f"{k.title()}: {v}",ln=1)
    pdf.ln(5)
    pdf.set_font("Helvetica","B",14); pdf.cell(0,8,"LLM Recommendation:",ln=1)
    pdf.set_font("Helvetica",size=12)
    for line in llm_msg.splitlines(): pdf.multi_cell(0,6,line)
    pdf.ln(5)
    if counts:
        fig,ax=plt.subplots()
        ax.bar(counts.keys(), counts.values(), color="#61a5c2")
        ax.set_title("Current Stock Levels")
        tmp_chart = create_temp_png(fig); plt.close(fig)
        pdf.image(tmp_chart, w=180); pdf.ln(5)
        fig,ax=plt.subplots()
        ax.pie(counts.values(), labels=counts.keys(), autopct="%1.1f%%")
        ax.set_title("Stock Composition")
        tmp_pie = create_temp_png(fig); plt.close(fig)
        pdf.image(tmp_pie, w=120); pdf.ln(5)
    if os.path.exists(FRAME_FILE):
        pdf.add_page()
        pdf.set_font("Helvetica","B",14); pdf.cell(0,8,"Last Detection Frame:",ln=1)
        pdf.image(FRAME_FILE, w=180)
    return pdf.output(dest="S").encode("latin1")

# ─── LOAD DATA ─────────────────────────────────────────────────────────────
counts = json.load(open(COUNT_FILE)) if os.path.exists(COUNT_FILE) else {}
if os.path.exists(HIST_FILE):
    try:
        history_df = pd.read_csv(HIST_FILE, parse_dates=["timestamp"], on_bad_lines="skip", engine="python")
    except:
        history_df = pd.DataFrame()
else:
    history_df = pd.DataFrame()

# ─── UI TABS ─────────────────────────────────────────────────────────────────
tabs = st.tabs(["Dashboard","Inventory","Reports"])

# Dashboard
with tabs[0]:
    st.header("Executive Dashboard")
    total = sum(counts.values())
    out0  = sum(v==0 for v in counts.values())
    low0  = sum(0<v<=min_threshold for v in counts.values())
    days  = (datetime.now() - (history_df.timestamp.min() if not history_df.empty else datetime.now())).days
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total SKUs", total)
    c2.metric("Out-of-Stock", out0, delta=f"{out0} ↑")
    c3.metric(f"Low-Stock (≤{min_threshold})", low0, delta=f"{low0} ↑")
    c4.metric("Days Running", days)
    st.markdown("---")
    for sku,qty in counts.items():
        if qty==0: st.toast(f"🚨 {sku.title()} OUT OF STOCK")
        elif qty<=min_threshold: st.toast(f"⚠️ {sku.title()} low: {qty}")
    if st.button("🔄 Refresh AI Recommendation"):
        st.session_state.llm_msg = call_llm(counts, history_df)
    st.info(st.session_state.llm_msg or "Click to generate advice…")

# Inventory
with tabs[1]:
    st.header("Live Inventory")
    cols = st.columns(len(counts) or 1)
    for i,(sku,qty) in enumerate(counts.items()):
        bg = "#e9f7ef" if qty>min_threshold else "#fff3cd" if qty>0 else "#f8d7da"
        status = "✅ OK" if qty>min_threshold else ("⚠️ Low" if qty>0 else "❌ Out")
        with cols[i]:
            st.markdown(f"<div style='background:{bg};padding:1rem;border-radius:0.5rem;text-align:center;'>"
                        f"<h4>{sku.title()}</h4><h2>{qty}</h2><small>{status}</small></div>", unsafe_allow_html=True)
    st.subheader("Last Detection Frame")
    if os.path.exists(FRAME_FILE):
        ts = open(TS_FILE).read(); st.image(FRAME_FILE, caption=ts, use_column_width=True)
    else:
        st.info("Waiting for first capture…")

# Reports
with tabs[2]:
    st.header("Export Reports")
    dcol,wcol = st.columns(2)
    with dcol:
        if st.button("📥 Download Daily PDF"):
            pdf_bytes = create_pdf("daily", counts, st.session_state.llm_msg, history_df)
            st.download_button("Download Daily", data=pdf_bytes,
                               file_name=f"daily_report_{datetime.now().date()}.pdf",
                               mime="application/pdf")
    with wcol:
        if st.button("📥 Download Weekly PDF"):
            pdf_bytes = create_pdf("weekly", counts, st.session_state.llm_msg, history_df)
            st.download_button("Download Weekly", data=pdf_bytes,
                               file_name=f"weekly_summary_{datetime.now().date()}.pdf",
                               mime="application/pdf")
