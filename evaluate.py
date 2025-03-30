import os
import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from math import pi
from torch import nn
from models.model import UNet
from config import *

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# -------- Load trained model --------
model = UNet().to(DEVICE)
checkpoint = torch.load('./models/checkpoint/best_model_fold_1.pt', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("✅ Model loaded.")

# -------- Ellipse circumference estimator --------
def ellipse_head_circumference(major_axis, minor_axis):
    a, b = major_axis / 2, minor_axis / 2
    return pi * (3*(a + b) - np.sqrt((3*a + b)*(a + 3*b)))  # Ramanujan approx

# -------- Dice score --------
def dice_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    return 2. * intersection / (pred.sum() + gt.sum() + 1e-6)

# -------- Load CSV --------
df = pd.read_csv('./src/test_set_pixe_size_new.csv')

# -------- Paths --------
data_dir = 'src/generated_test_set/'
os.makedirs('visuals', exist_ok=True)

# -------- Evaluation Loop --------
predicted_hcs = []
gt_hcs = []
dice_scores = []

for _, row in df.iterrows():
    fname = row['filename'].replace('.png', '')  # remove extension
    pixel_size = row['pixel size(mm)']
    gt_hc = row['head circumference (mm)']

    img_path = os.path.join(data_dir, f"{fname}.npy")
    mask_path = os.path.join(data_dir, f"{fname}_Annotation.npy")

    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        print(f"⚠️ Missing file: {fname}")
        continue

    # Load input image and GT mask
    image = np.load(img_path)  # shape: (H, W)
    gt_mask = np.load(mask_path)

    # Preprocess input
    image_input = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)  # shape: [1, 1, H, W]

    # Predict
    with torch.no_grad():
        pred_mask = model(image_input)
        pred_mask = (pred_mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255  # binary mask

    # Fit ellipse to prediction
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue
    largest_contour = max(contours, key=cv2.contourArea)
    if len(largest_contour) < 5:
        continue

    ellipse = cv2.fitEllipse(largest_contour)
    _, axes, _ = ellipse
    major_axis, minor_axis = axes
    hc_pixels = ellipse_head_circumference(major_axis, minor_axis)
    hc_mm = hc_pixels * pixel_size

    # Dice
    gt_mask_bin = (gt_mask > 0).astype(np.uint8) * 255
    gt_mask_bin = cv2.resize(gt_mask_bin, pred_mask.shape[::-1])
    dice = dice_score(pred_mask, gt_mask_bin)

    # Store results
    predicted_hcs.append(hc_mm)
    gt_hcs.append(gt_hc)
    dice_scores.append(dice)
    print(f"{fname} | Dice: {dice:.4f} | HC_pred: {hc_mm:.2f} mm | HC_gt: {gt_hc:.2f} mm")

    # -------- Visualization --------
    norm_image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    ellipse_mask = pred_mask.copy()
    ellipse_img = cv2.cvtColor(ellipse_mask, cv2.COLOR_GRAY2BGR)
    if len(largest_contour) >= 5:
        cv2.ellipse(ellipse_img, ellipse, (0, 255, 0), 2)  # green ellipse

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(norm_image, cmap='gray')
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(pred_mask, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(gt_mask_bin, cmap='gray')
    plt.title('GT Mask')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(ellipse_img)
    plt.title('Pred + Ellipse')
    plt.axis('off')

    plt.suptitle(f"{fname} | Dice: {dice:.4f} | HC: {hc_mm:.2f}mm", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join("visuals", f"{fname}_result.png"), dpi=150)
    plt.close()

# -------- Final results --------
mae = np.mean(np.abs(np.array(predicted_hcs) - np.array(gt_hcs)))
mean_dice = np.mean(dice_scores)

print(f"\n Mean Absolute Error (HC): {mae:.2f} mm")
print(f" Mean Dice Score: {mean_dice:.4f}")
