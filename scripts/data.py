import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import cv2
import torch
import numpy as np

def preprocess_mask(mask):
    # Convert mask from 0-255 to binary 0 and 1
    binary_mask = (mask > 0).astype(np.uint8)  # Threshold at 0; adjust if needed (e.g., == 255)
    return binary_mask

class CustomUltrasoundDataset(Dataset):
    def __init__(self, annnootation_file, image_dir,target_size=(540, 800), transform=None):
        self.hc_df = pd.read_csv(annnootation_file)
        self.image_dir = image_dir
        self.target_size = target_size
        self.transform = transform

    def __len__(self):
        return len(self.hc_df)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir,self.hc_df.iloc[idx,0])
        segmentation_mask_path = os.path.join(self.image_dir,f"{self.hc_df.iloc[idx,0].split('.')[0]}_Annotation.{self.hc_df.iloc[idx,0].split('.')[1]}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not os.path.exists(segmentation_mask_path):
            raise FileNotFoundError(f"Mask file not found: {segmentation_mask_path}")
            
        # Load images
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(segmentation_mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize images to target size
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)  # Use nearest for masks
        mask = preprocess_mask(mask)

        # Convert to tensors and normalize
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize image
        mask = torch.tensor(mask, dtype=torch.long)  # Keep as integer labels

        if self.transform:
            pass
            # image = self.transform(image)

        return image, mask


# if __name__ == "__main__":
#     dataset = CustomUltrasoundDataset(
#         annnootation_file="./src/training_set_pixel_size_and_HC.csv",
#         image_dir="./src/training_set"
#     )
#     print(dataset[0])

# annonation_file = "./src/training_set_pixel_size_and_HC.csv"
# img_dir = "./src/training_set"
# c = CustomUltrasoundDataset(annonation_file, img_dir)
# print(c[3])