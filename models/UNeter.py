import torch
import torch.nn as nn
import torch.nn.functional as F

class OrangeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(OrangeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv1(x)))

class GreenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GreenBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.deconv(x)

class BlueBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BlueBlock, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class GreyBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GreyBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv1(x)

class UNETR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.patch_embedding = nn.Linear(config["patch_height"] * config["patch_width"] * config["num_channels"], config["hidden_dim"])
        self.positions_embeddings = nn.Embedding(config["num_patches"], config["hidden_dim"])

        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=config["hidden_dim"], nhead=12, dim_feedforward=config["mlp_dim"],
                                       dropout=config["dropout_rate"], activation="gelu", batch_first=True)
            for _ in range(config["num_layers"])
        ])

        # Decoder blocks (predefined for GPU compatibility)
        self.z9_d1 = BlueBlock(config["hidden_dim"], 512)
        self.z12_d1 = GreenBlock(config["hidden_dim"], 512)
        self.z9z12_conv1 = OrangeBlock(1024, 512)
        self.z9z12_conv2 = OrangeBlock(512, 512)

        self.z9z12_up = GreenBlock(512, 256)
        self.z6_d1 = BlueBlock(config["hidden_dim"], 256)
        self.z6_d2 = BlueBlock(256, 256)
        self.z6z9z12_conv1 = OrangeBlock(512, 256)
        self.z6z9z12_conv2 = OrangeBlock(256, 256)

        self.z6z9z12_up = GreenBlock(256, 128)
        self.z3_d1 = BlueBlock(config["hidden_dim"], 128)
        self.z3_d2 = BlueBlock(128, 128)
        self.z3_d3 = BlueBlock(128, 128)
        self.z3z6z9z12_conv1 = OrangeBlock(256, 128)
        self.z3z6z9z12_conv2 = OrangeBlock(128, 128)

        self.z3z6z9z12_up = GreenBlock(128, 64)
        self.z0_d1 = OrangeBlock(1, 64)
        self.z0_d2 = OrangeBlock(64, 64)
        self.z0z3z6z9z12_conv1 = OrangeBlock(128, 64)
        self.z0z3z6z9z12_conv2 = OrangeBlock(64, 64)

        self.output_layer = GreyBlock(64, 1)

    def forward(self, inputs):
        batch_size, channels, height, width = inputs.shape
        device = inputs.device
        patch_height, patch_width = self.config["patch_height"], self.config["patch_width"]
        num_patches = self.config["num_patches"]

        patches = inputs.unfold(2, patch_height, patch_height).unfold(3, patch_width, patch_width)
        patches = patches.contiguous().view(batch_size, channels, -1, patch_height * patch_width)
        patches = patches.permute(0, 2, 1, 3).reshape(batch_size, num_patches, -1)

        patch_embedding = self.patch_embedding(patches)
        positions = torch.arange(0, num_patches, device=device)
        pos_embed = self.positions_embeddings(positions).unsqueeze(0).to(patch_embedding.device)
        x = patch_embedding + pos_embed

        connection_map = [3, 6, 9, 12]
        feature_map = []
        for i, layer in enumerate(self.encoder_layers, 1):
            x = layer(x)
            if i in connection_map:
                feature_map.append(x)

        z3, z6, z9, z12 = feature_map
        h, w = height // patch_height, width // patch_width
        hidden_shape = (batch_size, self.config["hidden_dim"], h, w)

        def reshape(z):
            return z.permute(0, 2, 1).contiguous().view(hidden_shape)

        z3, z6, z9, z12 = map(reshape, [z3, z6, z9, z12])

        z9_d1 = self.z9_d1(z9)
        z12_d1 = self.z12_d1(z12)
        z9z12 = torch.cat([z9_d1, z12_d1], dim=1)
        z9z12 = self.z9z12_conv1(z9z12)
        z9z12 = self.z9z12_conv2(z9z12)

        z9z12_up = self.z9z12_up(z9z12)
        z6_d = self.z6_d1(z6)
        z6_d = self.z6_d2(z6_d)
        z6z9z12 = torch.cat([z6_d, z9z12_up], dim=1)
        z6z9z12 = self.z6z9z12_conv1(z6z9z12)
        z6z9z12 = self.z6z9z12_conv2(z6z9z12)

        z6z9z12_up = self.z6z9z12_up(z6z9z12)
        z3_d = self.z3_d1(z3)
        z3_d = self.z3_d2(z3_d)
        z3_d = self.z3_d3(z3_d)
        z3z6z9z12 = torch.cat([z3_d, z6z9z12_up], dim=1)
        z3z6z9z12 = self.z3z6z9z12_conv1(z3z6z9z12)
        z3z6z9z12 = self.z3z6z9z12_conv2(z3z6z9z12)

        z3z6z9z12_up = self.z3z6z9z12_up(z3z6z9z12)
        z0_resized = F.interpolate(inputs, size=z3z6z9z12_up.shape[2:], mode="bilinear", align_corners=False)
        z0_d1 = self.z0_d1(z0_resized)
        z0_d1 = self.z0_d2(z0_d1)

        z0z3z6z9z12 = torch.cat([z0_d1, z3z6z9z12_up], dim=1)
        z0z3z6z9z12 = self.z0z3z6z9z12_conv1(z0z3z6z9z12)
        z0z3z6z9z12 = self.z0z3z6z9z12_conv2(z0z3z6z9z12)

        output = self.output_layer(z0z3z6z9z12)
        output = F.interpolate(output, size=(self.config["image_height"], self.config["image_width"]), mode="bilinear", align_corners=False)
        return output

# Config
config = {
    "image_height": 540,
    "image_width": 800,
    "num_channels": 1,
    "patch_height": 10,
    "patch_width": 10,
    "num_patches": (540 * 800) // (10 * 10),  # 4320
    "num_layers": 12,
    "hidden_dim": 768,
    "mlp_dim": 3072,
    "dropout_rate": 0.3,
}

# Ensure this is the only code that runs
if __name__ == "__main__":
    device = torch.device("cuda")
    model = UNETR(config).to(device=device)  # Move model to GPU at instantiation
    image = torch.randn(1, 1, 540, 800).to(device=device)  # Create and move image to GPU in one step
    print(f"Image Tensor Shape: {image.shape}")
    
    # For debugging: Show patch shape
    batch_size = image.shape[0]  # 8
    patches = image.unfold(2, 10, 10).unfold(3, 10, 10)  # [8, 1, 54, 80, 10, 10]
    patches = patches.contiguous().view(batch_size, 1, -1, 10 * 10)  # [8, 1, 4320, 100]
    patches = patches.permute(0, 2, 1, 3).reshape(batch_size, 4320, 100)  # [8, 4320, 100]
    print(f"Patch Tensor Shape: {patches.shape}")

    # Pass raw image to model
    output = model(image)  # [8, 1, 540, 800], no need to move output to device again
    print(f"Output Shape: {output.shape}")