Object Detection in Adverse Weather Conditions using DETR
Overview
This project explores a domain adaptation technique — Fourier Domain Adaptation (FDA) — integrated with the Detection Transformer (DETR) architecture to perform robust object detection under challenging weather conditions such as fog, rain, and snow. Adverse weather environments significantly affect visibility and scene clarity, which limits the accuracy of traditional object detectors. This work demonstrates how FDA can help bridge the domain gap without retraining, making DETR more resilient for real-world deployment.

Objectives
Evaluate the baseline DETR model under clear and adverse weather conditions

Apply Fourier Domain Adaptation to reduce domain shift

Improve detection performance (mAP) under degraded visibility

System Design
FDA: Swaps low-frequency components in the Fourier domain to align style/illumination of target images

DETR: Anchor-free transformer-based object detection framework with global attention

Combined Pipeline: FDA-preprocessed images are fed to a frozen DETR for detection

Specifications
Domain Adaptation: Fourier Domain Adaptation (FDA)

Object Detector: Detection Transformer (DETR) with ResNet-50 backbone

Framework: PyTorch

Image Resolution: 224x224

Datasets:

DAWN (adverse weather images)

BDD100K (diverse driving scenarios)

Training Strategy: Inference-only (no fine-tuning required)

Outputs: Bounding boxes and class labels

Implementation Steps
Fourier Transform: Transform the source (adverse weather) and reference (clear weather) images to frequency domain

Amplitude Swapping: Swap low-frequency components of adverse images with those from clear images

Inverse Transform: Convert the adapted image back to spatial domain

Object Detection: Feed the adapted image to the DETR model

Output: Obtain and visualize predictions

Results
Baseline DETR

mAP@50: 35.94%

mAP: 21.30%

FDA + DETR

mAP@50: 38.45%

mAP: 23.49%

FDA improved detection confidence and localization in adverse weather with no retraining of DETR. It also maintained consistent performance in clear conditions, demonstrating lightweight and robust adaptation.

Key Findings
FDA bridges the domain gap by aligning low-frequency features while preserving object structures

Works as a plug-and-play preprocessing step for any pre-trained detector

Lightweight, no expensive retraining required

Highly tunable with blending factors and frequency area size

Future Scope
Optimize FDA and DETR for real-time embedded applications

Handle complex weather scenarios (e.g., foggy night, extreme snowstorms)

Integrate end-to-end learnable adaptation instead of static FDA

Test on larger and more diverse datasets

References
Carion et al., End-to-End Object Detection with Transformers, 2020

Wang et al., Fourier Domain Adaptation for Grape Leaf Disease Identification, 2024

J. Iqbal et al., FogAdapt: Self-Supervised Domain Adaptation for Foggy Images, 2022

Gharatappeh et al., Weather-Aware Object Detection Transformer, 2025

Contributors
Vineet Desai [USN: 01FE22BEC176]

Tushar Pyati [USN: 01FE22BEC177]

K L Bhargava Prasad [USN: 01FE22BEC206]

Rohit Kumar [USN: 01FE22BEC224]

Guided by Prof. Preeti Pillai
School of Electronics and Communication Engineering
KLE Technological University, Hubballi

