import torch
import torch.nn as nn

class ImprovedFeatureToImageDecoder(nn.Module):
    def __init__(self, feature_dim=512, img_size=64):
        super().__init__()
        self.img_size = img_size

        # Feature processing with residual
        self.feature_proc = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
        )

        # Reshape to spatial
        start_size = img_size // 8  # 8x8 for 64x64 output
        self.reshape = nn.Linear(2048, 256 * start_size * start_size)

        # Progressive upsampling with refinement convs
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8→16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 16→32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 32→64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )

        # Attention
        self.attention = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Sigmoid(),
        )

        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 3, 7, 1, 3),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.feature_proc(x)
        start_size = self.img_size // 8
        x = self.reshape(x).view(-1, 256, start_size, start_size)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        attn = self.attention(x)
        x = x * attn
        return self.final(x)
