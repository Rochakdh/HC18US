import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

# -----------------------------------------------
# Double Convolution Block: (Conv => BN => ReLU) * 2
# Used throughout the network for feature extraction
# -----------------------------------------------
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),  # preserves spatial size
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# -----------------------------------------------
# Initial Convolution Block: wraps double_conv
# -----------------------------------------------
class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)

# -----------------------------------------------
# Downsampling Block: MaxPool followed by double_conv
# Used in the encoder path
# -----------------------------------------------
class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),  # reduces spatial resolution by half
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.mpconv(x)

# -----------------------------------------------
# Upsampling Block: 
# Upsamples (learnable or bilinear) + concatenates encoder features + double_conv
# -----------------------------------------------
class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            # Non-learnable upsampling using bilinear interpolation
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # Learnable upsampling via transposed convolution
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)

        # After concatenation, input channels will be doubled, hence `in_ch`
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        # x1: output from previous layer, x2: corresponding encoder output
        x1 = self.up(x1)

        # Compute spatial differences and pad x1 to match x2 size
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # Concatenate along channel dimension and apply double conv
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# -----------------------------------------------
# Final Output Convolution Layer: 
# Reduces channels to the number of target classes
# -----------------------------------------------
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# -----------------------------------------------
# U-Net Architecture
# Encoder: inc -> down1 -> down2 -> down3 -> down4
# Decoder: up1 -> up2 -> up3 -> up4
# -----------------------------------------------
class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(UNet, self).__init__()

        # Encoder path
        self.inc = inconv(n_channels, 64)        # Initial conv
        self.down1 = down(64, 128)               # Downsample 1
        self.down2 = down(128, 256)              # Downsample 2
        self.down3 = down(256, 512)              # Downsample 3
        self.down4 = down(512, 512)              # Downsample 4 (bottleneck)

        # Decoder path with learnable upsampling (ConvTranspose2d)
        self.up1 = up(1024, 256, bilinear=False) # Upsample 1
        self.up2 = up(512, 128, bilinear=False)  # Upsample 2
        self.up3 = up(256, 64, bilinear=False)   # Upsample 3
        self.up4 = up(128, 64, bilinear=False)   # Upsample 4

        # Final output layer
        self.outc = outconv(64, n_classes)

        # Dropout for regularization to reduce overfitting
        self.dropout = torch.nn.Dropout2d(0.5)

    def forward(self, x):
        x = x.float()                    # Ensure input is float
        x1 = self.inc(x)                # Encoder block 1
        x2 = self.down1(x1)             # Encoder block 2
        x3 = self.down2(x2)             # Encoder block 3
        x4 = self.down3(x3)             # Encoder block 4
        x5 = self.down4(x4)             # Bottleneck

        x = self.up1(x5, x4)            # Decoder block 1
        x = self.up2(x, x3)             # Decoder block 2
        x = self.dropout(x)             # Dropout applied here
        x = self.up3(x, x2)             # Decoder block 3
        x = self.up4(x, x1)             # Decoder block 4

        x = self.outc(x)                # Output layer (logits or before activation)
        return x                        # Can apply torch.sigmoid(x) externally for binary seg.


# # Instantiate the model
# model = UNet(n_channels=1, n_classes=1)

# # Test with input of shape (B, C, H, W) = (2, 1, 540, 800)
# x = torch.randn(2, 1, 540, 800)
# output = model(x)
# print(output)
# print("Output shape:", output.shape)  # Should be (2, 1, 540, 800)
