import torch
import torch.nn.functional as F

# Dice Loss for segmentation tasks: measures overlap between predicted and ground truth masks
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)  # Convert raw logits to probabilities
    pred = pred.contiguous()    # Ensure tensor memory layout is consistent
    target = target.contiguous()

    # Calculate intersection and union for Dice score
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    # Dice coefficient
    dice = (2. * intersection + smooth) / (union + smooth)

    # Dice loss = 1 - Dice coefficient
    return 1 - dice.mean()

# Focal Loss: Focuses on hard examples by down-weighting easy ones
def focal_loss(pred, target, alpha=0.8, gamma=2.0):
    pred = torch.sigmoid(pred)  # Convert raw logits to probabilities

    # Standard binary cross-entropy
    bce = F.binary_cross_entropy(pred, target, reduction='none')

    # pt = probability of correct class
    pt = torch.exp(-bce)

    # Focal loss formula
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()

# Combined Loss: Weighted sum of BCE, Dice, and Focal losses
def bce_dice_focal_loss(pred, target, bce_weight=0.3, dice_weight=0.5, focal_weight=0.2):
    """
    Computes a composite loss combining:
    - Binary Cross Entropy with Logits (for pixel-wise classification)
    - Dice Loss (for spatial overlap)
    - Focal Loss (for hard example mining)

    Args:
        pred: Raw logits from model (B, 1, H, W)
        target: Ground truth masks (B, H, W) or (B, 1, H, W)
        bce_weight: Weight for BCE loss
        dice_weight: Weight for Dice loss
        focal_weight: Weight for Focal loss

    Returns:
        Combined weighted loss
    """
    # If target is missing channel dimension, add it
    if target.ndim == 3:
        target = target.unsqueeze(1)
    
    target = target.float()  # Ensure target is float for loss computations

    # Compute individual loss components
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dsc = dice_loss(pred, target)
    focal = focal_loss(pred, target)

    # Weighted combination
    loss = bce_weight * bce + dice_weight * dsc + focal_weight * focal
    return loss
