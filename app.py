import os, sqlite3
from datetime import datetime
from pathlib import Path
import cv2, numpy as np, streamlit as st
from visionqc.detector import inspect_image

DB = os.getenv("VISIONQC_DB", "visionqc.db")
MODEL = os.getenv("VISIONQC_MODEL", "models/best.onnx")
st.set_page_config(page_title="VisionQC", page_icon="🔎", layout="wide")
st.title("VisionQC")
st.caption("PCB visual inspection demo")
with sqlite3.connect(DB) as conn:
    conn.execute("CREATE TABLE IF NOT EXISTS inspections (id INTEGER PRIMARY KEY, created_at TEXT, filename TEXT, status TEXT, defect_count INTEGER, mode TEXT)")
uploaded = st.file_uploader("Upload a PCB image", type=["png","jpg","jpeg","bmp"])
demo = st.checkbox("Use generated demo PCB", value=uploaded is None)
image, filename = None, ""
if uploaded:
    image = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    filename = uploaded.name
elif demo:
    image = np.full((480,720,3), 235, dtype=np.uint8)
    cv2.rectangle(image,(70,55),(650,425),(40,120,40),3)
    for x in range(120,630,70):
        for y in range(100,400,65):
            cv2.circle(image,(x,y),8,(170,150,70),2)
    cv2.line(image,(100,230),(610,230),(40,120,40),3)
    filename = "generated_demo_pcb.png"
if image is not None:
    if image is None:
        st.error("Unable to decode image.")
        st.stop()
    result = inspect_image(image, MODEL)
    a,b = st.columns([3,2])
    with a:
        st.image(cv2.cvtColor(result["annotated"],cv2.COLOR_BGR2RGB), use_container_width=True)
    with b:
        st.metric("Candidate regions",len(result["detections"]))
        st.metric("Mode","ONNX model" if result["model_used"] else "Demo heuristic")
        if result["detections"]: st.dataframe(result["detections"],use_container_width=True)
        st.warning("Demo heuristic results are illustrative, not validated manufacturing decisions.")
    status = "REVIEW" if result["detections"] else "PASS"
    with sqlite3.connect(DB) as conn:
        conn.execute("INSERT INTO inspections(created_at,filename,status,defect_count,mode) VALUES(?,?,?,?,?)",
          (datetime.now().isoformat(timespec="seconds"),filename,status,len(result["detections"]),result["mode"]))
        rows = conn.execute("SELECT id,created_at,filename,status,defect_count FROM inspections ORDER BY id DESC LIMIT 20").fetchall()
    st.subheader("Recent inspection history")
    st.dataframe(rows,use_container_width=True)
