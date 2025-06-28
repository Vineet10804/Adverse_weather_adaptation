import os
import warnings
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor
import time
from pathlib import Path
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from random import sample, choice

# ---------------------- Suppress Warnings ----------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("transformers").setLevel(logging.ERROR)

# ---------------------- Configuration ----------------------
model_name = "facebook/detr-resnet-50"
dataset_dir = "D:\\COLLEGE\\6th_SEM\\Minor_project\\dawn"
labels_dir = os.path.join(dataset_dir, "labels")
images_dir = os.path.join(dataset_dir, "images")
reference_images_dir = "D:\\COLLEGE\\6th_SEM\\Minor_project\\bdd10k\\images\\train"
max_reference_images = 50
class_names = ["person", "bicycle", "MotorCycle", "Car", "Bus", "Truck", "Train"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- Load Model ----------------------
model = DetrForObjectDetection.from_pretrained(model_name).to(device)
processor = DetrImageProcessor.from_pretrained(model_name)

# ---------------------- Load Reference Images ----------------------
def load_reference_images(folder_path, limit=None):
    image_paths = list(Path(folder_path).glob("*.jpg"))
    if not image_paths:
        print(f"No reference images found in {folder_path}")
        return []
    if limit:
        image_paths = sample(image_paths, min(limit, len(image_paths)))
    images = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
    print(f"Loaded {len(images)} reference images from {folder_path}")
    return images

reference_images = load_reference_images(reference_images_dir, max_reference_images)
if not reference_images:
    print("No reference images available. Exiting.")
    exit()

# ---------------------- Helper Functions ----------------------
def apply_fda(source_img, target_img, beta=0.001):
    source = np.array(source_img, dtype=np.float32) / 255.0
    target = np.array(Image.fromarray(np.array(target_img)).resize(source_img.size))
    target = np.array(target, dtype=np.float32) / 255.0

    s_fft = np.fft.fft2(source, axes=(0, 1))
    t_fft = np.fft.fft2(target, axes=(0, 1))
    s_fft_shifted = np.fft.fftshift(s_fft, axes=(0, 1))
    t_fft_shifted = np.fft.fftshift(t_fft, axes=(0, 1))

    s_amp = np.abs(s_fft_shifted)
    s_phase = np.angle(s_fft_shifted)
    t_amp = np.abs(t_fft_shifted)

    h, w, _ = source.shape
    h_crop, w_crop = int(h * beta), int(w * beta)
    h_center, w_center = h // 2, w // 2

    mask = np.zeros((h, w), dtype=np.float32)
    mask[h_center - h_crop//2:h_center + h_crop//2, w_center - w_crop//2:w_center + w_crop//2] = 1

    new_amp = s_amp * (1 - mask[..., np.newaxis]) + t_amp * mask[..., np.newaxis]

    new_fft = new_amp * np.exp(1j * s_phase)
    new_fft_shifted = np.fft.ifftshift(new_fft, axes=(0, 1))
    new_image = np.abs(np.fft.ifft2(new_fft_shifted, axes=(0, 1)))

    new_image = (new_image - new_image.min()) / (new_image.max() - new_image.min())
    return Image.fromarray((new_image * 255).astype(np.uint8))

def load_yolo_labels(label_path, img_width, img_height):
    boxes, labels = [], []
    if not os.path.exists(label_path):
        return boxes, labels

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            try:
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                x_min = max(0, (cx - w/2) * img_width)
                y_min = max(0, (cy - h/2) * img_height)
                x_max = min(img_width, (cx + w/2) * img_width)
                y_max = min(img_height, (cy + h/2) * img_height)
                if x_max > x_min and y_max > y_min:
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id)
            except Exception as e:
                print(f"Error processing line: {line.strip()} - {str(e)}")
    return boxes, labels

def process_detr_outputs(outputs, img_w, img_h, confidence_threshold):
    scores = outputs.logits.softmax(-1)[0, :, :-1]
    keep = scores.max(-1).values > confidence_threshold

    boxes = outputs.pred_boxes[0, keep].cpu().numpy()
    labels = scores[keep].argmax(dim=1).cpu().numpy()
    confidences = scores[keep].max(dim=1).values.cpu().numpy()

    abs_boxes = []
    for box in boxes:
        cx, cy, w, h = box
        x_min = max(0, (cx - w/2) * img_w)
        y_min = max(0, (cy - h/2) * img_h)
        x_max = min(img_w, (cx + w/2) * img_w)
        y_max = min(img_h, (cy + h/2) * img_h)
        abs_boxes.append([x_min, y_min, x_max, y_max])

    return abs_boxes, labels, confidences

# ---------------------- Evaluation Pipeline ----------------------
def evaluate_with_fda(conf_threshold=0.5, beta=0.001):
    print(f"\nEvaluating with confidence_threshold = {conf_threshold:.2f}, beta = {beta}...")
    
    all_gt_boxes, all_gt_labels = [], []
    all_pred_boxes, all_pred_scores, all_pred_labels = [], [], []

    image_files = list(Path(images_dir).glob("*.jpg"))
    if not image_files:
        print("No images found in directory.")
        return

    start_eval_time = time.time()

    for i, img_path in enumerate(image_files):
        source_image = Image.open(img_path).convert("RGB")
        img_w, img_h = source_image.size

        reference_image = choice(reference_images)
        adapted_image = apply_fda(source_image, reference_image, beta)

        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        gt_boxes, gt_labels = load_yolo_labels(label_path, img_w, img_h)

        inputs = processor(images=adapted_image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        pred_boxes, pred_labels, pred_scores = process_detr_outputs(
            outputs, img_w, img_h, conf_threshold)

        all_gt_boxes.append(gt_boxes)
        all_gt_labels.append(gt_labels)
        all_pred_boxes.append(pred_boxes)
        all_pred_scores.append(pred_scores)
        all_pred_labels.append(pred_labels)

        if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
            print(f"Processed {i+1}/{len(image_files)} images")

    end_eval_time = time.time()

    print("Computing mAP metrics...")
    targets, preds = [], []

    for gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels in zip(
        all_gt_boxes, all_gt_labels, all_pred_boxes, all_pred_scores, all_pred_labels):

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

    print(f"Evaluation time: {end_eval_time - start_eval_time:.2f}s")
    print(f"mAP: {final_map['map']:.4f}, mAP@50: {final_map['map_50']:.4f}, mAP@75: {final_map['map_75']:.4f}")

    return final_map

# ---------------------- Run Evaluation ----------------------
if __name__ == "__main__":
    thresholds = [0.1]
    beta_value = 0.001
    results = []

    for th in thresholds:
        map_result = evaluate_with_fda(conf_threshold=th, beta=beta_value)
        results.append((th, map_result['map'].item()))

    print("\nSummary of mAP for different thresholds:")
    for th, m in results:
        print(f"Threshold {th:.2f} => mAP: {m:.4f}")

    best = max(results, key=lambda x: x[1])
    print(f"\n✅ Best confidence threshold: {best[0]:.2f} with mAP = {best[1]:.4f}")
