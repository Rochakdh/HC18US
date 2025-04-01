import torch
import torch.nn.functional as F

# bce_loss = torch.nn.BCELoss()

# def dice_loss(pred, target, smooth=1e-6):
#     """
#     Computes the Dice Loss for binary segmentation.
#     Args:
#         pred: Tensor of predictions (batch_size, 1, H, W).
#         target: Tensor of ground truth (batch_size, H, W) or (batch_size, 1, H, W).
#         smooth: Smoothing factor to avoid division by zero.
#     Returns:
#         Scalar Dice Loss.
#     """
#     # Apply sigmoid to convert logits to probabilities
#     pred = torch.sigmoid(pred)

#     # Ensure target has the same shape as pred
#     # if target.dim() == 3:  # (B, H, W)
#     #     target = target.unsqueeze(1)  # Convert to (B, 1, H, W) Or do below to squeeze (1) 
    
#     pred = pred.squeeze(1)
#     # Ensure shapes match
#     assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape}, target {target.shape}"

#     # Calculate intersection and union
#     intersection = (pred * target).sum(dim=(1,2))
#     union = pred.sum(dim=(1,2)) + target.sum(dim=(1,2))
    
#     # Compute Dice Coefficient
#     dice = (2. * intersection + smooth) / (union + intersection + smooth)
    
#     # Return Dice Loss
#     return 1 - dice.mean()


# -----------x--------------------x-----------------------------


# def compute_bce_loss(pred, target):
#     return bce_loss(pred, target)


# def dice_loss(pred, target, smooth=1e-6):
#     pred = pred.contiguous()
#     target = target.contiguous()

#     intersection = (pred * target).sum(dim=(2, 3))
#     union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

#     dice = (2. * intersection + smooth) / (union + smooth)
#     loss = 1 - dice
#     return loss.mean()

# def bce_dice_loss(pred, target, bce_weight=0.3):
#     # Ensure target shape matches pred shape: [B, 1, H, W]
#     if target.ndim == 3:
#         target = target.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

#     # Ensure target is float for BCE
#     target = target.float()

#     bce = compute_bce_loss(pred, target)
#     dice = dice_loss(pred, target)
#     return bce_weight * bce + (1 - bce_weight) * dice


# import torch
# import torch.nn.functional as F

# def dice_loss(pred, target, smooth=1e-6):
#     pred = torch.sigmoid(pred)  # Apply sigmoid inside Dice
#     pred = pred.contiguous()
#     target = target.contiguous()

#     intersection = (pred * target).sum(dim=(2, 3))
#     union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

#     dice = (2. * intersection + smooth) / (union + smooth)
#     loss = 1 - dice
#     return loss.mean()

# def bce_dice_loss(pred, target, bce_weight=0.5):
#     # Ensure target shape is [B, 1, H, W] and float
#     if target.ndim == 3:
#         target = target.unsqueeze(1)
#     target = target.float()

#     bce = F.binary_cross_entropy_with_logits(pred, target)
#     dice = dice_loss(pred, target)
#     return bce_weight * bce + (1 - bce_weight) * dice



import torch
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)  # Convert logits to probabilities
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def focal_loss(pred, target, alpha=0.8, gamma=2.0):
    pred = torch.sigmoid(pred)
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()


def bce_dice_focal_loss(pred, target, bce_weight=0.3, dice_weight=0.5, focal_weight=0.2):
    """
    Combined loss: BCEWithLogits + Dice + Focal
    """
    if target.ndim == 3:
        target = target.unsqueeze(1)
    target = target.float()

    bce = F.binary_cross_entropy_with_logits(pred, target)
    dsc = dice_loss(pred, target)
    focal = focal_loss(pred, target)

    loss = bce_weight * bce + dice_weight * dsc + focal_weight * focal
    return loss