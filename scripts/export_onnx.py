import argparse
from ultralytics import YOLO
p=argparse.ArgumentParser(); p.add_argument("--weights",required=True); p.add_argument("--imgsz",type=int,default=640)
a=p.parse_args()
print("Exported model:",YOLO(a.weights).export(format="onnx",imgsz=a.imgsz,dynamic=False))
