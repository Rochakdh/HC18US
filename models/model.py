import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv2D => BN => ReLU => Dropout) * 2"""
    def __init__(self, in_channels, out_channels, dropout_prob=0.3):
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
    def __init__(self, in_channels=1, out_channels=1, dropout_prob=0.3):
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
        """Resize input, process through U-Net, then resize output"""
        orig_size = x.shape[-2:]  # Save original (H, W)
        x = F.interpolate(x, size=(544, 800), mode='bilinear', align_corners=False)

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

        # Decoder path
        x = self.up4(x5)
        x = self.dropout4(x)  # Apply dropout before concatenation
        x = torch.cat([x, x4], dim=1)
        x = self.conv4(x)

        x = self.up3(x)
        x = self.dropout3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv3(x)

        x = self.up2(x)
        x = self.dropout2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        x = self.dropout1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv1(x)

        x = self.final(x)
        x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=False)  # Resize back to original (540, 800)

        return x

# # Instantiate the model
# model = UNet(in_channels=1, out_channels=1, dropout_prob=0.3)

# # Test with input of shape (B, C, H, W) = (2, 1, 540, 800)
# x = torch.randn(2, 1, 540, 800)
# output = model(x)
# print("Output shape:", output.shape)  # Should be (2, 1, 540, 800)
