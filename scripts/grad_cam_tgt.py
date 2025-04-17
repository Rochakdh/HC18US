import torch
import numpy as np

class SemanticSegmentationTarget:
    """
    This class defines a custom target function for visualization methods like Grad-CAM
    in semantic segmentation tasks. It extracts the activation values at regions specified
    by a given binary mask (e.g., for a specific class or object).
    """
    
    def __init__(self, mask):
        """
        Args:
            mask (np.ndarray): A binary or soft mask (H, W) indicating the target region.
        """
        # Convert the NumPy mask to a PyTorch tensor with float32 precision
        self.mask = torch.from_numpy(mask.astype(np.float32))

        # Move the mask to GPU if available for compatibility with model outputs
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        """
        Applies the target mask to the model output for computing weighted activation.
        
        Args:
            model_output (torch.Tensor): The raw model output (e.g., logits or features),
                                         typically of shape (1, C, H, W)
        
        Returns:
            torch.Tensor: A scalar value obtained by summing masked activations
        """
        # Ensure the model output tensor is on the same device as the mask
        model_output = model_output.to(self.mask.device)

        # Element-wise multiplication between the output and mask, followed by sum
        return (model_output[0] * self.mask).sum()
