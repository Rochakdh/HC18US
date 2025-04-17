import os
from torch.utils.data import Dataset
import pandas as pd
import cv2
import torch
import numpy as np
import torchvision.transforms.functional as TF
from torchvision import transforms



def pad_image_and_mask(image_tensor, mask_tensor, target_width=800, target_height=540): #TODO: Only run genreating after data_set
    """Pads both image and mask tensors to target size while keeping them centered."""
    _, h, w = image_tensor.shape  # Get current dimensions (C, H, W)
    pad_left = (target_width - w) // 2
    pad_right = target_width - w - pad_left
    pad_top = (target_height - h) // 2
    pad_bottom = target_height - h - pad_top

    # Pad image (3 channels) and mask (1 channel) separately
    image_padded = TF.pad(image_tensor, (pad_left, pad_top, pad_right, pad_bottom), fill=0)  # Pad with 0 (black)
    mask_padded = TF.pad(mask_tensor, (pad_left, pad_top, pad_right, pad_bottom), fill=0)  # Pad with 0 (background)

    return image_padded, mask_padded


class CustomUltrasoundDataset(Dataset):
    def __init__(self, preprocessed_dir, annotation_file, transform=None, augment=False):
        self.hc_df = pd.read_csv(annotation_file)
        self.preprocessed_dir = preprocessed_dir  # Load from preprocessed directory
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.hc_df) * (2 if self.augment else 1)

    def __getitem__(self, idx):
        """ Returns Image and it corresponding mask for training purpose"""
        original_idx = idx // (2 if self.augment else 1)

        image_name = self.hc_df.iloc[original_idx, 0]
        image_npy_path = os.path.join(self.preprocessed_dir, f"{image_name.split('.')[0]}.npy")
        mask_npy_path = os.path.join(self.preprocessed_dir, f"{image_name.split('.')[0]}_Annotation.npy")

        if not os.path.exists(image_npy_path) or not os.path.exists(mask_npy_path):
            raise FileNotFoundError(f"File not found: {image_npy_path} or {mask_npy_path}")

        # Load preprocessed NumPy arrays
        image = np.load(image_npy_path).astype(np.float32)  # Already normalized [0,1]
        mask = np.load(mask_npy_path).astype(np.uint8)  # Already binary

        # Apply augmentation if enabled
        if self.augment and idx % 2 == 1 and self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
            image,mask = pad_image_and_mask(image,mask)
            image = torch.tensor(image, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.long)
        else:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(mask, dtype=torch.long)
        return image, mask



# def preprocess_mask(mask):
#     """ Convert mask from 0-255 to binary 0 and 1 """
#     return (mask > 0).astype(np.uint8)  # Threshold at 0

# def fill_mask_holes(mask):
#     # Convert mask to binary (ensure only 0 and 255)
#     _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

#     # Create an inverse mask where holes are the background
#     mask_floodfill = binary_mask.copy()
#     h, w = mask.shape

#     # Create a mask for flood fill (2 pixels larger than input)
#     floodfill_mask = np.zeros((h+2, w+2), np.uint8)

#     # Flood fill from top-left corner (assuming background is black)
#     cv2.floodFill(mask_floodfill, floodfill_mask, (0,0), 255)

#     # Invert the flood-filled mask
#     mask_filled_holes = cv2.bitwise_not(mask_floodfill)

#     # Combine with original mask (fill holes without altering outer structures)
#     filled_mask = cv2.bitwise_or(binary_mask, mask_filled_holes)

#     return filled_mask