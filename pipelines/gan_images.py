# ================================
# Improved Kinship Feature → Image Reconstruction
# ================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg19
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# 1. Dataset (with data augmentation)
# -------------------------------
class FeatureImageDataset(Dataset):
    def __init__(self, feats_npy, images_dir, img_size=64):
        self.features = np.load(feats_npy)

        # Find all images in subdirectories
        image_files = []
        for root, dirs, files in os.walk(images_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    rel_path = os.path.relpath(os.path.join(root, f), images_dir)
                    image_files.append(rel_path)
        image_files = sorted(image_files)

        count = min(len(self.features), len(image_files))
        self.features = self.features[:count]
        self.images = image_files[:count]
        self.dir = images_dir

        # Enhanced transforms with augmentation
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.5),  # Data augmentation
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3)
        ])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = torch.FloatTensor(self.features[idx])
        img_path = os.path.join(self.dir, self.images[idx])
        img = Image.open(img_path).convert('RGB')
        return feat, self.transform(img)

# -------------------------------
# 2. Enhanced Decoder with Skip Connections & Attention
# -------------------------------
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

        # Progressive upsampling with skip connections
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8→16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 1, 1),  # Refinement conv
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 16→32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 32→64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )

        # Attention mechanism - Output channels changed from 16 to 32
        self.attention = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), # Changed output channels to 32
            nn.Sigmoid()
        )

        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 3, 7, 1, 3),  # Larger kernel for smoothing
            nn.Tanh()
        )

    def forward(self, x):
        # Process features
        x = self.feature_proc(x)

        # Reshape to spatial
        start_size = self.img_size // 8
        x = self.reshape(x).view(-1, 256, start_size, start_size)

        # Progressive upsampling
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        # Apply attention
        attention = self.attention(x)
        x = x * attention

        # Final output
        return self.final(x)

# -------------------------------
# 3. Discriminator for Adversarial Loss
# -------------------------------
class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------
# 4. Enhanced Loss Functions
# -------------------------------
class ImprovedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

        # Load VGG for perceptual loss
        self.vgg = vgg19(weights="IMAGENET1K_V1").features[:16].eval().to(DEVICE)
        for p in self.vgg.parameters():
            p.requires_grad = False

    def perceptual_loss(self, fake, real):
        fake_features = self.vgg(fake)
        real_features = self.vgg(real)
        return self.mse(fake_features, real_features)

    def adversarial_loss(self, pred, is_real):
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return self.bce(pred, target)

# -------------------------------
# 5. Training Setup
# -------------------------------
base_dir = "/content/drive/MyDrive/KinFaceW-II"
features_path = "/content/normalized_features.npy"
images_dir = os.path.join(base_dir, "images")

IMG_SIZE = 64
dataset = FeatureImageDataset(features_path, images_dir, IMG_SIZE)
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

# Models
generator = ImprovedFeatureToImageDecoder(img_size=IMG_SIZE).to(DEVICE)
discriminator = PatchDiscriminator().to(DEVICE)

# Optimizers with different learning rates
opt_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Schedulers for learning rate decay
scheduler_G = optim.lr_scheduler.StepLR(opt_G, step_size=20, gamma=0.5)
scheduler_D = optim.lr_scheduler.StepLR(opt_D, step_size=20, gamma=0.5)

criterion = ImprovedLoss()

# -------------------------------
# 6. Enhanced Training Loop
# -------------------------------
EPOCHS = 100  # More epochs for better results
CHECKPOINT_DIR = "/content/decoder_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Training history
history = {'G_loss': [], 'D_loss': [], 'L1_loss': [], 'VGG_loss': []}

print(f"Starting training with {len(dataset)} samples...")

for epoch in range(EPOCHS):
    G_losses, D_losses, L1_losses, VGG_losses = [], [], [], []

    for i, (feats, real_imgs) in enumerate(loader):
        feats, real_imgs = feats.to(DEVICE), real_imgs.to(DEVICE)
        batch_size = real_imgs.size(0)

        # ===============================
        # Train Discriminator
        # ===============================
        opt_D.zero_grad()

        # Real images
        pred_real = discriminator(real_imgs)
        loss_D_real = criterion.adversarial_loss(pred_real, True)

        # Fake images
        with torch.no_grad():
            fake_imgs = generator(feats)
        pred_fake = discriminator(fake_imgs.detach())
        loss_D_fake = criterion.adversarial_loss(pred_fake, False)

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        opt_D.step()

        # ===============================
        # Train Generator
        # ===============================
        opt_G.zero_grad()

        fake_imgs = generator(feats)

        # Adversarial loss
        pred_fake = discriminator(fake_imgs)
        loss_G_adv = criterion.adversarial_loss(pred_fake, True)

        # L1 loss
        loss_L1 = criterion.l1(fake_imgs, real_imgs)

        # Perceptual loss
        loss_VGG = criterion.perceptual_loss(fake_imgs, real_imgs)

        # Combined generator loss
        loss_G = loss_G_adv + 100 * loss_L1 + 10 * loss_VGG
        loss_G.backward()
        opt_G.step()

        # Store losses
        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())
        L1_losses.append(loss_L1.item())
        VGG_losses.append(loss_VGG.item())

    # Update learning rates
    scheduler_G.step()
    scheduler_D.step()

    # Average losses
    avg_G = np.mean(G_losses)
    avg_D = np.mean(D_losses)
    avg_L1 = np.mean(L1_losses)
    avg_VGG = np.mean(VGG_losses)

    history['G_loss'].append(avg_G)
    history['D_loss'].append(avg_D)
    history['L1_loss'].append(avg_L1)
    history['VGG_loss'].append(avg_VGG)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  G_loss: {avg_G:.4f}, D_loss: {avg_D:.4f}")
    print(f"  L1_loss: {avg_L1:.4f}, VGG_loss: {avg_VGG:.4f}")

    # Save checkpoints and generate samples
    if (epoch + 1) % 10 == 0:
        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'opt_G': opt_G.state_dict(),
            'opt_D': opt_D.state_dict(),
            'epoch': epoch,
            'history': history
        }, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{epoch+1}.pth"))

        # Generate sample images
        generator.eval()
        with torch.no_grad():
            sample_feats = feats[:4]  # First 4 samples
            sample_real = real_imgs[:4]
            sample_fake = generator(sample_feats)

            # Save sample grid
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            for j in range(4):
                # Real images
                real_img = (sample_real[j].cpu() + 1) / 2
                axes[0, j].imshow(real_img.permute(1, 2, 0))
                axes[0, j].set_title('Real')
                axes[0, j].axis('off')

                # Generated images
                fake_img = (sample_fake[j].cpu() + 1) / 2
                axes[1, j].imshow(fake_img.permute(1, 2, 0).clamp(0, 1))
                axes[1, j].set_title('Generated')
                axes[1, j].axis('off')

            plt.tight_layout()
            plt.savefig(f"/content/samples_epoch{epoch+1}.png", dpi=150)
            plt.close()
        generator.train()

# Final save
torch.save({
    'generator': generator.state_dict(),
    'discriminator': discriminator.state_dict(),
    'history': history
}, os.path.join(CHECKPOINT_DIR, "final_model.pth"))

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['G_loss'], label='Generator')
plt.plot(history['D_loss'], label='Discriminator')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Adversarial Losses')

plt.subplot(1, 2, 2)
plt.plot(history['L1_loss'], label='L1 Loss')
plt.plot(history['VGG_loss'], label='VGG Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Reconstruction Losses')

plt.tight_layout()
plt.savefig('/content/training_curves.png', dpi=150)
plt.show()

print("Enhanced training completed! ✅")