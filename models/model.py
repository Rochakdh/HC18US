import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv2D => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """Resize input, process through U-Net, then resize output"""
        orig_size = x.shape[-2:]  # Save original (H, W)
        x = F.interpolate(x, size=(544, 800), mode='bilinear', align_corners=False)

        x1 = self.down1(x)
        x2 = self.maxpool(x1)
        x2 = self.down2(x2)
        x3 = self.maxpool(x2)
        x3 = self.down3(x3)
        x4 = self.maxpool(x3)
        x4 = self.down4(x4)
        x5 = self.maxpool(x4)
        x5 = self.bottleneck(x5)

        x = self.up4(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv4(x)

        x = self.up3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv3(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv1(x)

        x = self.final(x)
        x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=False)  # Resize back to original (540, 800)

        return x

# Instantiate the model
model = UNet(in_channels=1, out_channels=1)

# Test with input of shape (B, C, H, W) = (2, 1, 540, 800)
x = torch.randn(2, 1, 540, 800)
output = model(x)
print("Output shape:", output.shape)  # Should be (2, 1, 540, 800)
