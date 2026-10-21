import argparse
from ultralytics import YOLO
p=argparse.ArgumentParser()
p.add_argument("--data",default="data.yaml"); p.add_argument("--weights",default="yolov8n.pt")
p.add_argument("--epochs",type=int,default=50); p.add_argument("--imgsz",type=int,default=640)
a=p.parse_args()
model=YOLO(a.weights)
model.train(data=a.data,epochs=a.epochs,imgsz=a.imgsz)
print("Review held-out validation metrics before making performance claims.")
