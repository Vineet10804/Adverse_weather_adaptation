import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor
import time
import pandas as pd
import os

# ---------------------- Configuration ----------------------
model_name = "facebook/detr-resnet-50"
image_path = "D:\\COLLEGE\\6th_SEM\\Minor_project\\dawn\\images\\haze-050.jpg"   # Replace with your image path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- Load Model ----------------------
model = DetrForObjectDetection.from_pretrained(model_name).to(device)
processor = DetrImageProcessor.from_pretrained(model_name)

# ---------------------- Load Image ----------------------
assert os.path.exists(image_path), f"Image not found: {image_path}"
image = Image.open(image_path).convert("RGB")

# ---------------------- Preprocess and Inference ----------------------
inputs = processor(images=image, return_tensors="pt").to(device)
confidence_threshold = 0.90
start_time = time.time()
with torch.no_grad():
    outputs = model(**inputs)
inference_time = time.time() - start_time

# ---------------------- Decode Predictions ----------------------
img_w, img_h = image.size
scores = outputs.logits.softmax(-1)[0, :, :-1]
keep = scores.max(-1).values > confidence_threshold

# ---------------------- Collect Performance Metrics ----------------------
detections = keep.sum().item()
confidences = scores[keep].max(dim=1).values.tolist()
mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
max_conf = max(confidences) if confidences else 0.0
detected_classes = [model.config.id2label[idx.item()] for idx in scores[keep].argmax(dim=1)]

# ---------------------- Visualization ----------------------
fig, ax = plt.subplots(1, figsize=(10, 8))
ax.imshow(image)

if detections == 0:
    print("⚠️ No objects detected with high confidence.")
else:
    print(f"✅ Detected objects ({detections}):")
    for score, box in zip(scores[keep], outputs.pred_boxes[0, keep]):
        label_id = score.argmax().item()
        label = model.config.id2label[label_id]
        bbox = box.tolist()

        # Convert relative to absolute box coordinates
        x_center, y_center, width, height = bbox
        x_min = (x_center - width / 2) * img_w
        y_min = (y_center - height / 2) * img_h
        width *= img_w
        height *= img_h

        # Draw bounding box
        rect = patches.Rectangle(
            (x_min, y_min), width, height, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
        text_label = f"{label}: {score.max().item():.2f}"
        ax.text(x_min, y_min - 5, text_label, fontsize=12, color="red", weight="bold")

        print(f" - {label:15} | Confidence: {score.max().item():.4f} | BBox: {bbox}")

plt.axis("off")
plt.tight_layout()
plt.show()

# ---------------------- Performance Matrix ----------------------
metrics = {
    "Detections": [detections],
    "Mean Confidence": [mean_conf],
    "Max Confidence": [max_conf],
    "Inference Time (s)": [inference_time],
    "Detected Classes": [", ".join(detected_classes)],
}

df = pd.DataFrame(metrics, index=["Original Image"])
print("\n📊 Performance Summary (Original Image):")
print(df)

# Optionally save
df.to_csv("DETR_original_image_performance.csv", index=True)

# ---------------------- Conclusion ----------------------
print("\n📝 Conclusion:")
print("This baseline run shows object detection performance without domain adaptation.")
print("Use this result to compare with FDA or other adaptation methods under domain shift.")
