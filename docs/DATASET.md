# Dataset preparation
Expected YOLO dataset layout:
```
datasets/pcb/
  images/train/
  images/val/
  labels/train/
  labels/val/
```
Each label file contains `class_id x_center y_center width height` with normalized coordinates. Keep the class IDs aligned with `data.yaml`; separate validation images by board/capture session to reduce data leakage.
