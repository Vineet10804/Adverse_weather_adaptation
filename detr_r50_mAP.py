import os
import warnings
import logging

# ---------------------- Suppress Warnings ----------------------
# Suppress Hugging Face transformers warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress TensorFlow logs (for oneDNN and other INFO logs)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("transformers").setLevel(logging.ERROR)

# ---------------------- Imports ----------------------
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor
import time
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import average_precision_score

# ---------------------- Configuration ----------------------
model_name = "facebook/detr-resnet-50"
image_path = "D:\\COLLEGE\\6th_SEM\\Minor_project\\foggy3.jpg"  # Single image for visualization
dataset_dir = "D:\\COLLEGE\\6th_SEM\\Minor_project\\dawn"  # Path to your dataset
labels_dir = os.path.join(dataset_dir, "labels")  # Labels in YOLO format
images_dir = os.path.join(dataset_dir, "images")  # Images folder
class_names = ["person", "bicycle", "MotorCycle", "Car", "Bus","Truck","Train"]  # Add classes as per your dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
confidence_threshold = 0.5

# ---------------------- Load Model ----------------------
model = DetrForObjectDetection.from_pretrained(model_name).to(device)
processor = DetrImageProcessor.from_pretrained(model_name)

# ---------------------- Helper Functions ----------------------
def load_yolo_labels(label_path, img_width, img_height):
    boxes, labels = [], []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            x_min = (cx - w / 2) * img_width
            y_min = (cy - h / 2) * img_height
            x_max = (cx + w / 2) * img_width
            y_max = (cy + h / 2) * img_height
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_id)
    return boxes, labels

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

# ---------------------- Evaluation Loop ----------------------
all_gt_boxes, all_gt_labels = [], []
all_pred_boxes, all_pred_scores, all_pred_labels = [], [], []
start_eval_time = time.time()

image_files = list(Path(images_dir).glob("*.jpg"))
for img_path in image_files:
    image = Image.open(img_path).convert("RGB")
    img_w, img_h = image.size
    label_path = Path(labels_dir) / (img_path.stem + ".txt")
    gt_boxes, gt_labels = load_yolo_labels(label_path, img_w, img_h)
    
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = outputs.logits.softmax(-1)[0, :, :-1]
    keep = scores.max(-1).values > confidence_threshold
    boxes = outputs.pred_boxes[0, keep].cpu().numpy()
    labels = scores[keep].argmax(dim=1).cpu().numpy()
    confidences = scores[keep].max(dim=1).values.cpu().numpy()
    
    abs_boxes = []
    for box in boxes:
        cx, cy, w, h = box
        x_min = (cx - w/2) * img_w
        y_min = (cy - h/2) * img_h
        x_max = (cx + w/2) * img_w
        y_max = (cy + h/2) * img_h
        abs_boxes.append([x_min, y_min, x_max, y_max])
    
    all_gt_boxes.append(gt_boxes)
    all_gt_labels.append(gt_labels)
    all_pred_boxes.append(abs_boxes)
    all_pred_scores.append(confidences)
    all_pred_labels.append(labels)

end_eval_time = time.time()

# ---------------------- Compute mAP ----------------------
from torchmetrics.detection.mean_ap import MeanAveragePrecision

targets = []
preds = []
for gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels in zip(all_gt_boxes, all_gt_labels, all_pred_boxes, all_pred_scores, all_pred_labels):
    targets.append({
        "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
        "labels": torch.tensor(gt_labels, dtype=torch.int64)
    })
    preds.append({
        "boxes": torch.tensor(pred_boxes, dtype=torch.float32),
        "scores": torch.tensor(pred_scores, dtype=torch.float32),
        "labels": torch.tensor(pred_labels, dtype=torch.int64)
    })

metric = MeanAveragePrecision()
metric.update(preds, targets)
final_map = metric.compute()

# ---------------------- Report ----------------------
print("\n mAP Evaluation on Dawn Dataset:")
for k, v in final_map.items():
    if torch.is_tensor(v):
        if v.numel() == 1:  # Single scalar
            print(f"{k}: {v.item():.4f}")
        else:  # Tensor with multiple values (e.g. per-class mAP)
            # Format output for vector metrics (e.g., list all values or summarize)
            values = v.cpu().numpy()
            print(f"{k}: [{', '.join(f'{x:.4f}' for x in values)}]")
    else:
        print(f"{k}: {v}")


print(f"\n Evaluation Time: {end_eval_time - start_eval_time:.2f}s")
