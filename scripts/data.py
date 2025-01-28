import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd

class CustomUltrasoundDataset(Dataset):
    def __init__(self, annnootation_file, image_dir):
        self.hc_df = pd.read_csv(annnootation_file)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.hc_df)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir,self.hc_df.iloc[idx,0])
        segmentation_mask = os.path.join(self.image_dir,f"{self.hc_df.iloc[idx,0].split(".")[0]}_Annonation.{self.hc_df.iloc[idx,0].split(".")[1]}")
        return image_path, segmentation_mask
    

annonation_file = "./src/training_set_pixel_size_and_HC.csv"
img_dir = "./src/training_set"
c = CustomUltrasoundDataset(annonation_file, img_dir)
print(c[3])