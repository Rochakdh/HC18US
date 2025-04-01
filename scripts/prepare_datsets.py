import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Config
ANNOTATION_FILE = "./src/training_set_pixel_size_and_HC.csv"
IMAGE_DIR = "./src/training_set/"
OUTPUT_BASE_DIR = "./src/"
TRAIN_DIR = os.path.join(OUTPUT_BASE_DIR, "generated_training_set")
TEST_DIR = os.path.join(OUTPUT_BASE_DIR, "generated_test_set")

# Ensure directories exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Load annotation file
hc_df = pd.read_csv(ANNOTATION_FILE)

# Split into train/test
train_df, test_df = train_test_split(hc_df, test_size=0.2, random_state=42)

# Save split CSVs
train_df.to_csv(os.path.join(OUTPUT_BASE_DIR, "train_generated.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_BASE_DIR, "test_generated.csv"), index=False)

# Resize target
TARGET_SIZE = (800, 540)

def fill_mask_holes(mask):
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask_floodfill = binary_mask.copy()
    h, w = mask.shape
    floodfill_mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(mask_floodfill, floodfill_mask, (0,0), 255)
    mask_filled_holes = cv2.bitwise_not(mask_floodfill)
    return cv2.bitwise_or(binary_mask, mask_filled_holes)

def preprocess_mask(mask):
    return (mask > 127).astype(np.uint8)

def process_and_save(df, save_dir):
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_name = row.iloc[0]
        image_path = os.path.join(IMAGE_DIR, image_name)
        mask_path = os.path.join(IMAGE_DIR, f"{image_name.split('.')[0]}_Annotation.{image_name.split('.')[1]}")
        
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Skipping missing files: {image_name}")
            continue

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

        mask = fill_mask_holes(mask)
        mask = preprocess_mask(mask)

        min_val, max_val = image.min(), image.max()
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image)

        image_npy_path = os.path.join(save_dir, f"{image_name.split('.')[0]}.npy")
        mask_npy_path = os.path.join(save_dir, f"{image_name.split('.')[0]}_Annotation.npy")
        np.save(image_npy_path, image)
        np.save(mask_npy_path, mask)

# Process and save both sets
process_and_save(train_df, TRAIN_DIR)
process_and_save(test_df, TEST_DIR)

print(f"All processing complete. Files saved to: {OUTPUT_BASE_DIR}")
