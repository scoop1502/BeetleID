"""
Model locations for BeetleID's own copy of the detection / alignment / order pipeline.
All files are in BeetleID/models (see models/MODELS.md).
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # BeetleID
MODELS = os.path.join(ROOT, "models")

DETECTOR_RTDETR = os.path.join(MODELS, "detector_rtdetr_v3.pt")
DETECTOR_YOLO = os.path.join(MODELS, "detector_yolo11n.pt")
KEYPOINTS_CPM = os.path.join(MODELS, "keypoints_cpm_epoch240.pth")
CLASSIFIER = os.path.join(MODELS, "order_classifier_resnet18.pth")       # 21 orders, used to find beetles
CLASS_MAPPING = os.path.join(MODELS, "order_class_mapping.txt")
CLASSIFICATION_CODE = os.path.join(ROOT, "training", "classification")
