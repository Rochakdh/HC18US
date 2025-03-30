import torch
import numpy as np

class SemanticSegmentationTarget:
    def __init__(self, mask):
        self.mask = torch.from_numpy(mask.astype(np.float32))
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        # model_output shape: [1, H, W] for binary segmentation
        return (model_output[0] * self.mask).sum()