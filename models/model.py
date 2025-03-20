import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class DoubleConv(nn.Module):
    """(Conv2D => BN => ReLU => Dropout) * 2"""
    def __init__(self, in_channels, out_channels, dropout_prob=DROPOUT):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),  # Dropout after activation

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob)  # Second dropout
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dropout_prob=DROPOUT):
        super(UNet, self).__init__()

        # Encoder
        self.down1 = DoubleConv(in_channels, 64, dropout_prob)
        self.down2 = DoubleConv(64, 128, dropout_prob)
        self.down3 = DoubleConv(128, 256, dropout_prob)
        self.down4 = DoubleConv(256, 512, dropout_prob)

        self.bottleneck = DoubleConv(512, 1024, dropout_prob)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(1024, 512, dropout_prob)
        self.dropout4 = nn.Dropout2d(p=dropout_prob)  # Dropout before concatenation

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256, dropout_prob)
        self.dropout3 = nn.Dropout2d(p=dropout_prob)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128, dropout_prob)
        self.dropout2 = nn.Dropout2d(p=dropout_prob)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64, dropout_prob)
        self.dropout1 = nn.Dropout2d(p=dropout_prob)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # Save original dimensions
        orig_size = x.shape[-2:]  # (H, W), e.g., (540, 800)

        # Encoder path
        x1 = self.down1(x)
        x2 = self.maxpool(x1)
        x2 = self.down2(x2)
        x3 = self.maxpool(x2)
        x3 = self.down3(x3)
        x4 = self.maxpool(x3)
        x4 = self.down4(x4)
        x5 = self.maxpool(x4)
        x5 = self.bottleneck(x5)

        # Decoder path with cropping
        x = self.up4(x5)
        x = self.dropout4(x)
        Hx, Wx = x.shape[2], x.shape[3]
        x4_cropped = x4[:, :, :Hx, :Wx]  # Crop x4 to match x
        x = torch.cat([x, x4_cropped], dim=1)
        x = self.conv4(x)

        x = self.up3(x)
        x = self.dropout3(x)
        Hx, Wx = x.shape[2], x.shape[3]
        x3_cropped = x3[:, :, :Hx, :Wx]  # Crop x3 to match x
        x = torch.cat([x, x3_cropped], dim=1)
        x = self.conv3(x)

        x = self.up2(x)
        x = self.dropout2(x)
        Hx, Wx = x.shape[2], x.shape[3]
        x2_cropped = x2[:, :, :Hx, :Wx]  # Crop x2 to match x
        x = torch.cat([x, x2_cropped], dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        x = self.dropout1(x)
        Hx, Wx = x.shape[2], x.shape[3]
        x1_cropped = x1[:, :, :Hx, :Wx]  # Crop x1 to match x
        x = torch.cat([x, x1_cropped], dim=1)
        x = self.conv1(x)

        x = self.final(x)

        # Pad output to match original size
        current_H, current_W = x.shape[2], x.shape[3]
        pad_H = orig_size[0] - current_H
        pad_W = orig_size[1] - current_W
        if pad_H > 0 or pad_W > 0:
            x = F.pad(x, (0, pad_W, 0, pad_H), mode='replicate')

        return x

# # Instantiate the model
# model = UNet(in_channels=1, out_channels=1, dropout_prob=0.3)

# # Test with input of shape (B, C, H, W) = (2, 1, 540, 800)
# x = torch.randn(2, 1, 540, 800)
# output = model(x)
# print("Output shape:", output.shape)  # Should be (2, 1, 540, 800)
