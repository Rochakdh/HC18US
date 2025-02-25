import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1e-6):
    """
    Computes the Dice Loss for binary segmentation.
    Args:
        pred: Tensor of predictions (batch_size, 1, H, W).
        target: Tensor of ground truth (batch_size, H, W) or (batch_size, 1, H, W).
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Scalar Dice Loss.
    """
    # Apply sigmoid to convert logits to probabilities
    pred = torch.sigmoid(pred)

    # Ensure target has the same shape as pred
    # if target.dim() == 3:  # (B, H, W)
    #     target = target.unsqueeze(1)  # Convert to (B, 1, H, W) Or do below to squeeze (1) 
    
    pred = pred.squeeze(1)
    # Ensure shapes match
    assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape}, target {target.shape}"

    # Calculate intersection and union
    intersection = (pred * target).sum(dim=(1,2))
    union = pred.sum(dim=(1,2)) + target.sum(dim=(1,2))
    
    # Compute Dice Coefficient
    dice = (2. * intersection + smooth) / (union + intersection + smooth)
    
    # Return Dice Loss
    return 1 - dice.mean()
