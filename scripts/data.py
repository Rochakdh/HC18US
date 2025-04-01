import os
from torch.utils.data import Dataset
import pandas as pd
import cv2
import torch
import numpy as np
import torchvision.transforms.functional as TF
from torchvision import transforms

def preprocess_mask(mask):
    """ Convert mask from 0-255 to binary 0 and 1 """
    return (mask > 0).astype(np.uint8)  # Threshold at 0

def fill_mask_holes(mask):
    # Convert mask to binary (ensure only 0 and 255)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Create an inverse mask where holes are the background
    mask_floodfill = binary_mask.copy()
    h, w = mask.shape

    # Create a mask for flood fill (2 pixels larger than input)
    floodfill_mask = np.zeros((h+2, w+2), np.uint8)

    # Flood fill from top-left corner (assuming background is black)
    cv2.floodFill(mask_floodfill, floodfill_mask, (0,0), 255)

    # Invert the flood-filled mask
    mask_filled_holes = cv2.bitwise_not(mask_floodfill)

    # Combine with original mask (fill holes without altering outer structures)
    filled_mask = cv2.bitwise_or(binary_mask, mask_filled_holes)

    return filled_mask


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

# class CustomUltrasoundDataset(Dataset):
#     def __init__(self, annotation_file, image_dir, target_size=(540, 800), transform=None, augment=False):
#         self.hc_df = pd.read_csv(annotation_file)
#         self.image_dir = image_dir
#         self.target_size = target_size
#         self.transform = transform
#         self.augment = augment

#     def __len__(self):
#         return len(self.hc_df) * (2 if self.augment else 1)

#     def __getitem__(self, idx):
#         original_idx = idx // (2 if self.augment else 1)

#         image_path = os.path.join(self.image_dir, self.hc_df.iloc[original_idx, 0])
#         segmentation_mask_path = os.path.join(
#             self.image_dir,
#             f"{self.hc_df.iloc[original_idx, 0].split('.')[0]}_Annotation.{self.hc_df.iloc[original_idx, 0].split('.')[1]}"
#         )

#         if not os.path.exists(image_path):
#             raise FileNotFoundError(f"Image file not found: {image_path}")
#         if not os.path.exists(segmentation_mask_path):
#             raise FileNotFoundError(f"Mask file not found: {segmentation_mask_path}")

#         # Load images as NumPy arrays
#         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Shape (H, W)
#         mask = cv2.imread(segmentation_mask_path, cv2.IMREAD_GRAYSCALE)  # Shape (H, W)

#         # Ensure data is a NumPy array
#         image = np.array(image, dtype=np.float32)  # Convert to float for normalization
#         mask = np.array(mask, dtype=np.uint8)

#         # Resize images
#         image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
#         mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

#         # Fill holes in the mask
#         mask = fill_mask_holes(mask)

#         # Convert to binary mask
#         mask = preprocess_mask(mask)

#         # **Normalize Image Intensities (Min-Max Scaling)**
#         min_val, max_val = image.min(), image.max()
#         if max_val > min_val:
#             image = (image - min_val) / (max_val - min_val)  # Normalize to [0,1]
#         else:
#             image = np.zeros_like(image)  # Edge case: uniform images

#         # Apply augmentation if enabled
#         if self.augment and idx % 2 == 1 and self.transform:
#             augmented = self.transform(image=image, mask=mask) 
#             image, mask = augmented["image"], augmented["mask"]
#         else:
#             # Convert to tensors if no augmentation
#             image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
#             mask = torch.tensor(mask, dtype=torch.long)

#         return image, mask
    

class CustomUltrasoundDataset(Dataset):
    def __init__(self, preprocessed_dir, annotation_file, transform=None, augment=False):
        self.hc_df = pd.read_csv(annotation_file)
        self.preprocessed_dir = preprocessed_dir  # Load from preprocessed directory
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.hc_df) * (2 if self.augment else 1)

    def __getitem__(self, idx):
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
    
    # def apply_transform(self, image, mask=None):

    #     # Grayscale
    #     image = transforms.functional.to_grayscale(image)
    #     if self.train_mode:
    #         mask = transforms.functional.to_grayscale(mask)

    #     # Resize
    #     resize = transforms.Resize(size=self.input_size, interpolation=Image.BILINEAR)
    #     image = resize(image)
    #     if self.train_mode:
    #         mask = resize(mask)

    #     if self.train_mode:
    #         # Random affine
    #         random_aff = transforms.RandomAffine(
    #             degrees=0,
    #             translate=(0.1, 0.1),
    #             scale=(0.9, 1.1),
    #             resample=3,
    #             fillcolor=0,
    #         )
    #         ret = random_aff.get_params(
    #             random_aff.degrees,
    #             random_aff.translate,
    #             random_aff.scale,
    #             random_aff.shear,
    #             image.size,
    #         )
    #         image = F.affine(
    #             image,
    #             *ret,
    #             resample=random_aff.resample,
    #             fillcolor=random_aff.fillcolor
    #         )
    #         mask = F.affine(
    #             mask, *ret, resample=random_aff.resample, fillcolor=random_aff.fillcolor
    #         )

    #         # Random horizontal flipping
    #         if random.random() > 0.5:
    #             image = F.hflip(image)
    #             mask = F.hflip(mask)

    #         # Random vertical flipping
    #         if random.random() > 0.5:
    #             image = F.vflip(image)
    #             mask = F.vflip(mask)

    #     # Transform to tensor
    #     image = F.to_tensor(image)
    #     if self.train_mode:
    #         mask = F.to_tensor(mask)

    #     # Binarize mask
    #     if self.train_mode:
    #         mask = torch.where(
    #             mask > 0.1, torch.tensor(1.0), torch.tensor(0.0)
    #         )  # TODO: tune threshold

    #         return image, mask
    #     return image


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     import cv2
#     import os

#     # Load dataset
#     dataset = CustomUltrasoundDataset(
#         annotation_file="/workspace/HC18US/src/training_set_pixel_size_and_HC.csv",
#         preprocessed_dir="/workspace/HC18US/src/generated_training_set/",
#         transform=None,
#         augment=False
#     )

#     # Get the image and mask at index 3
#     image, mask = dataset[3]
#     print(image.shape)
#     print(mask.shape)

    # Convert tensors to NumPy arrays
    # image_np = (image.squeeze(0).numpy() * 255).astype(np.uint8)  # Scale back to 0-255
    # mask_np = mask.numpy().astype(np.uint8) * 255  # Ensure binary mask is in 0-255

    # # Define output directory
    # output_dir = "./dataset_test"
    # os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    # # Save image and mask using OpenCV
    # cv2.imwrite(os.path.join(output_dir, "image_3.png"), image_np)
    # cv2.imwrite(os.path.join(output_dir, "mask_3.png"), mask_np)

    # print(f"Saved images to {output_dir}")
