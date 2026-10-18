# VisionQC — Machine Vision Defect Inspection System

Starter portfolio project for PCB inspection using OpenCV, YOLOv8, ONNX Runtime, Streamlit and SQLite.

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app runs an illustrative OpenCV heuristic until a compatible trained ONNX model is provided. It is not production-validated. Metrics in the original project description are not verified by this scaffold.

## Train/export
Prepare a YOLO-format dataset under `datasets/pcb/`, check `data.yaml`, then run:
```bash
python scripts/train.py --data data.yaml --epochs 50
python scripts/export_onnx.py --weights runs/detect/train/weights/best.pt
```
Review `docs/EVALUATION.md` before reporting performance.
