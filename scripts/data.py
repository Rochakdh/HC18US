import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd

class CustomUltrasoundDataset(Dataset):
    def __init__(self, annnootation_file, image_dir):
        self.hc_df = pd.read_csv(annnootation_file)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.hc_info)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir,df.iloc[idx,0])
        segmentation_mask = os.path.join(self.image_dir,f"{self.hc_df.iloc[1,0].split(".")[0]}_Annonation.{self.hc_df.iloc[1,0].split(".")[1]}")
        return image, segmentation_mask


