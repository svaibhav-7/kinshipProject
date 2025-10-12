#!/usr/bin/env python3
"""
Phase 2: CycleGAN Age Normalization for Kinship Verification

This module implements age-invariant feature normalization using CycleGAN
with triplet loss for robust kinship verification across age gaps.

Key Components:
1. Feature-space CycleGAN for age domain translation
2. Triplet loss for identity preservation during translation
3. Age-invariant feature extraction and evaluation
4. Integration with Phase 1 feature extraction pipeline

Usage:
    python phase2_cyclegan_normalization.py --features-dir results_hybrid_features --epochs 200
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
import itertools
from typing import Dict, List, Tuple, Optional

# Add Phase 1 path for feature extraction
PHASE1_PATH = Path(__file__).parent
sys.path.append(str(PHASE1_PATH))

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {DEVICE}")

# ==================== DATASET ====================
class KinshipFeatureDataset(Dataset):
    """Dataset for loading Phase 1 extracted features with age/relationship labels"""
    
    def __init__(self, features_dir: str, feature_type: str = 'combined'):
        self.features_dir = Path(features_dir)
        self.feature_type = feature_type
        self.samples = []
        self.age_groups = {'child': [], 'adult': []}  # Simple age grouping
        
        self._load_samples()
        self._assign_age_groups()
    
    def _load_samples(self):
        """Load all feature files and metadata"""
        for category in ['father-son', 'father-dau', 'mother-son', 'mother-dau']:
            cat_path = self.features_dir / category
            if not cat_path.exists():
                continue
                
            for feat_file in cat_path.glob(f'*_{self.feature_type}_features.npy'):
                # Load features
                features = np.load(feat_file)
                
                # Load metadata
                meta_file = feat_file.with_name(feat_file.stem.replace(f'_{self.feature_type}_features', '_metadata.json'))
                metadata = {}
                if meta_file.exists():
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)
                
                self.samples.append({
                    'features': features,
                    'category': category,
                    'filename': feat_file.name,
                    'metadata': metadata,
                    'file_path': feat_file
                })
    
    def _assign_age_groups(self):
        """Assign samples to child/adult age groups based on filename patterns"""
        for i, sample in enumerate(self.samples):
            filename = sample['filename']
            # Simple heuristic: assume first number in pair is parent (adult), second is child
            # You may need to adjust this based on your dataset naming convention
            if '_1_' in filename or 'parent' in filename.lower():
                self.age_groups['adult'].append(i)
            else:
                self.age_groups['child'].append(i)
        
        print(f"[INFO] Loaded {len(self.age_groups['adult'])} adult samples, {len(self.age_groups['child'])} child samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = torch.FloatTensor(sample['features'])
        
        # Determine age group (0: child, 1: adult)
        age_label = 1 if idx in self.age_groups['adult'] else 0
        
        return {
            'features': features,
            'age_label': age_label,
            'category': sample['category'],
            'idx': idx
        }
    
    def get_samples_by_age(self, age_group: str) -> List[int]:
        """Get sample indices for specific age group"""
        return self.age_groups.get(age_group, [])

# ==================== CYCLEGAN ARCHITECTURE ====================
class FeatureGenerator(nn.Module):
    """Generator network for feature-space translation"""
    
    def __init__(self, feature_dim: int = 512, hidden_dims: List[int] = [256, 128, 256]):
        super().__init__()
        self.feature_dim = feature_dim
        
        layers = []
        prev_dim = feature_dim
        
        # Encoder
        for dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
        
        # Bottleneck with residual connection
        self.encoder = nn.Sequential(*layers)
        self.bottleneck = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        decoder_layers = []
        prev_dim = hidden_dims[-1]
        for dim in reversed(hidden_dims[:-1]):
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
        
        decoder_layers.extend([
            nn.Linear(prev_dim, feature_dim),
            nn.Tanh()  # Output in [-1, 1] range
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Residual connection
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        encoded = self.encoder(x)
        bottleneck = self.bottleneck(encoded)
        decoded = self.decoder(bottleneck)
        
        # Residual connection for stability
        output = decoded + self.residual_weight * x
        return output

class FeatureDiscriminator(nn.Module):
    """Discriminator network for feature authenticity"""
    
    def __init__(self, feature_dim: int = 512, hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = feature_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# ==================== LOSS FUNCTIONS ====================
class TripletLoss(nn.Module):
    """Triplet loss for identity preservation during age translation"""
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: Original features
            positive: Translated features (same identity, different age)
            negative: Different identity features
        """
        distance_positive = torch.pairwise_distance(anchor, positive, p=2)
        distance_negative = torch.pairwise_distance(anchor, negative, p=2)
        
        # We want distance_positive < distance_negative
        target = torch.ones_like(distance_positive)
        loss = self.ranking_loss(distance_negative, distance_positive, target)
        
        return loss

class CycleGANLoss(nn.Module):
    """Combined loss function for CycleGAN with triplet loss"""
    
    def __init__(self, lambda_cycle: float = 10.0, lambda_identity: float = 5.0, 
                 lambda_triplet: float = 2.0):
        super().__init__()
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.lambda_triplet = lambda_triplet
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.triplet_loss = TripletLoss()
    
    def adversarial_loss(self, discriminator_output, target_is_real: bool):
        """LSGan loss"""
        if target_is_real:
            target = torch.ones_like(discriminator_output)
        else:
            target = torch.zeros_like(discriminator_output)
        return self.mse_loss(discriminator_output, target)
    
    def cycle_consistency_loss(self, real, reconstructed):
        """L1 cycle consistency loss"""
        return self.l1_loss(real, reconstructed)
    
    def identity_loss(self, real, generated):
        """L1 identity preservation loss"""
        return self.l1_loss(real, generated)

# ==================== CYCLEGAN TRAINER ====================
class CycleGANTrainer:
    """Training manager for CycleGAN age normalization"""
    
    def __init__(self, feature_dim: int = 512, learning_rate: float = 2e-4, 
                 beta1: float = 0.5, beta2: float = 0.999):
        self.feature_dim = feature_dim
        
        # Networks
        self.G_child_to_adult = FeatureGenerator(feature_dim).to(DEVICE)
        self.G_adult_to_child = FeatureGenerator(feature_dim).to(DEVICE)
        self.D_adult = FeatureDiscriminator(feature_dim).to(DEVICE)
        self.D_child = FeatureDiscriminator(feature_dim).to(DEVICE)
        
        # Optimizers
        self.optimizer_G = optim.Adam(
            itertools.chain(self.G_child_to_adult.parameters(), self.G_adult_to_child.parameters()),
            lr=learning_rate, betas=(beta1, beta2)
        )
        self.optimizer_D_adult = optim.Adam(self.D_adult.parameters(), lr=learning_rate, betas=(beta1, beta2))
        self.optimizer_D_child = optim.Adam(self.D_child.parameters(), lr=learning_rate, betas=(beta1, beta2))
        
        # Loss function
        self.criterion = CycleGANLoss()
        
        # Training history
        self.train_history = {
            'G_loss': [], 'D_adult_loss': [], 'D_child_loss': [],
            'cycle_loss': [], 'identity_loss': [], 'triplet_loss': []
        }
    
    def train_step(self, adult_features, child_features):
        """Single training step"""
        batch_size = adult_features.size(0)
        
        # =============== Train Generators ===============
        self.optimizer_G.zero_grad()
        
        # Identity loss
        same_adult = self.G_child_to_adult(adult_features)
        same_child = self.G_adult_to_child(child_features)
        identity_loss_adult = self.criterion.identity_loss(adult_features, same_adult)
        identity_loss_child = self.criterion.identity_loss(child_features, same_child)
        identity_loss = (identity_loss_adult + identity_loss_child) / 2
        
        # Adversarial loss
        fake_adult = self.G_child_to_adult(child_features)
        fake_child = self.G_adult_to_child(adult_features)
        
        pred_fake_adult = self.D_adult(fake_adult)
        pred_fake_child = self.D_child(fake_child)
        
        adversarial_loss_adult = self.criterion.adversarial_loss(pred_fake_adult, True)
        adversarial_loss_child = self.criterion.adversarial_loss(pred_fake_child, True)
        adversarial_loss = (adversarial_loss_adult + adversarial_loss_child) / 2
        
        # Cycle consistency loss
        recovered_adult = self.G_child_to_adult(fake_child)
        recovered_child = self.G_adult_to_child(fake_adult)
        
        cycle_loss_adult = self.criterion.cycle_consistency_loss(adult_features, recovered_adult)
        cycle_loss_child = self.criterion.cycle_consistency_loss(child_features, recovered_child)
        cycle_loss = (cycle_loss_adult + cycle_loss_child) / 2
        
        # Triplet loss for identity preservation
        # Create negative samples by shuffling
        indices = torch.randperm(batch_size)
        negative_adult = adult_features[indices]
        negative_child = child_features[indices]
        
        triplet_loss_adult = self.criterion.triplet_loss(adult_features, fake_adult, negative_adult)
        triplet_loss_child = self.criterion.triplet_loss(child_features, fake_child, negative_child)
        triplet_loss = (triplet_loss_adult + triplet_loss_child) / 2
        
        # Total generator loss
        G_loss = (adversarial_loss + 
                  self.criterion.lambda_cycle * cycle_loss + 
                  self.criterion.lambda_identity * identity_loss +
                  self.criterion.lambda_triplet * triplet_loss)
        
        G_loss.backward()
        self.optimizer_G.step()
        
        # =============== Train Discriminators ===============
        # Train D_adult
        self.optimizer_D_adult.zero_grad()
        
        pred_real_adult = self.D_adult(adult_features)
        pred_fake_adult = self.D_adult(fake_adult.detach())
        
        D_adult_real_loss = self.criterion.adversarial_loss(pred_real_adult, True)
        D_adult_fake_loss = self.criterion.adversarial_loss(pred_fake_adult, False)
        D_adult_loss = (D_adult_real_loss + D_adult_fake_loss) / 2
        
        D_adult_loss.backward()
        self.optimizer_D_adult.step()
        
        # Train D_child
        self.optimizer_D_child.zero_grad()
        
        pred_real_child = self.D_child(child_features)
        pred_fake_child = self.D_child(fake_child.detach())
        
        D_child_real_loss = self.criterion.adversarial_loss(pred_real_child, True)
        D_child_fake_loss = self.criterion.adversarial_loss(pred_fake_child, False)
        D_child_loss = (D_child_real_loss + D_child_fake_loss) / 2
        
        D_child_loss.backward()
        self.optimizer_D_child.step()
        
        # Store losses
        losses = {
            'G_loss': G_loss.item(),
            'D_adult_loss': D_adult_loss.item(),
            'D_child_loss': D_child_loss.item(),
            'cycle_loss': cycle_loss.item(),
            'identity_loss': identity_loss.item(),
            'triplet_loss': triplet_loss.item()
        }
        
        return losses, fake_adult, fake_child
    
    def train(self, dataloader: DataLoader, epochs: int, save_dir: str):
        """Full training loop with enhanced logging"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Starting CycleGAN training for {epochs} epochs")

        for epoch in range(epochs):
            print(f"\n-- Starting epoch {epoch+1}/{epochs} --")  # New log

            epoch_losses = {key: 0.0 for key in self.train_history.keys()}
            num_batches = 0

            for batch_data in dataloader:
                features = batch_data['features'].to(DEVICE)
                age_labels = batch_data['age_label'].to(DEVICE)

                # Separate features by age
                adult_features = features[age_labels == 1]
                child_features = features[age_labels == 0]
                if len(adult_features)==0 or len(child_features)==0:
                    continue
                min_size = min(len(adult_features), len(child_features))
                adult_features = adult_features[:min_size]
                child_features = child_features[:min_size]

                losses, _, _ = self.train_step(adult_features, child_features)
                for k,v in losses.items():
                    epoch_losses[k] += v
                num_batches += 1

            # Average losses
            if num_batches>0:
                for k in epoch_losses:
                    epoch_losses[k] /= num_batches
                    self.train_history[k].append(epoch_losses[k])

            # Detailed log every epoch
            print(f"[Epoch {epoch+1}] G: {epoch_losses['G_loss']:.4f} | "
                  f"D_adult: {epoch_losses['D_adult_loss']:.4f} | "
                  f"D_child: {epoch_losses['D_child_loss']:.4f} | "
                  f"Cycle: {epoch_losses['cycle_loss']:.4f} | "
                  f"Id: {epoch_losses['identity_loss']:.4f} | "
                  f"Triplet: {epoch_losses['triplet_loss']:.4f}")

            # Save checkpoints every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(save_dir / f'checkpoint_epoch_{epoch+1}.pth', epoch)

        # Final save/check
        self.save_checkpoint(save_dir / 'final_model.pth', epochs-1)
        print(f"[INFO] Saved final checkpoint to {save_dir/'final_model.pth'}")  # Verify
        self.plot_training_history(save_dir / 'training_history.png')
        print(f"[INFO] Training history plot at {save_dir/'training_history.png'}")  # Verify

        print("[INFO] Training completed!")

    
    def save_checkpoint(self, filepath: Path, epoch: int):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'G_child_to_adult': self.G_child_to_adult.state_dict(),
            'G_adult_to_child': self.G_adult_to_child.state_dict(),
            'D_adult': self.D_adult.state_dict(),
            'D_child': self.D_child.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D_adult': self.optimizer_D_adult.state_dict(),
            'optimizer_D_child': self.optimizer_D_child.state_dict(),
            'train_history': self.train_history
        }
        torch.save(checkpoint, filepath)
        print(f"[INFO] Saved checkpoint: {filepath}")
    
    def load_checkpoint(self, filepath: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=DEVICE)
        
        self.G_child_to_adult.load_state_dict(checkpoint['G_child_to_adult'])
        self.G_adult_to_child.load_state_dict(checkpoint['G_adult_to_child'])
        self.D_adult.load_state_dict(checkpoint['D_adult'])
        self.D_child.load_state_dict(checkpoint['D_child'])
        
        self.train_history = checkpoint['train_history']
        
        print(f"[INFO] Loaded checkpoint: {filepath}")
        return checkpoint['epoch']
    
    def plot_training_history(self, save_path: Path):
        """Plot training loss curves"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (key, values) in enumerate(self.train_history.items()):
            if i >= 6:
                break
            axes[i].plot(values)
            axes[i].set_title(f'{key.replace("_", " ").title()}')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel('Loss')
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved training history plot: {save_path}")

# ==================== EVALUATION ====================
def evaluate_age_normalization(trainer: CycleGANTrainer, dataset: KinshipFeatureDataset, 
                               save_dir: str):
    """Evaluate age normalization quality and kinship verification improvement"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    trainer.G_child_to_adult.eval()
    trainer.G_adult_to_child.eval()
    
    # Collect all features
    all_features = []
    all_labels = []
    all_categories = []
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for batch_data in dataloader:
            features = batch_data['features'].to(DEVICE)
            age_labels = batch_data['age_label'].numpy()
            categories = batch_data['category']
            
            all_features.extend(features.cpu().numpy())
            all_labels.extend(age_labels)
            all_categories.extend(categories)
    
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    
    # Normalize features to adult domain
    normalized_features = []
    with torch.no_grad():
        for i, (feat, label) in enumerate(zip(all_features, all_labels)):
            feat_tensor = torch.FloatTensor(feat).unsqueeze(0).to(DEVICE)
            
            if label == 0:  # Child -> Adult
                normalized = trainer.G_child_to_adult(feat_tensor)
            else:  # Already adult
                normalized = feat_tensor
            
            normalized_features.append(normalized.cpu().numpy().squeeze())
    
    normalized_features = np.array(normalized_features)
    
    # Visualize with t-SNE
    print("[INFO] Generating t-SNE visualization...")
    
    # Original features
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    original_2d = tsne.fit_transform(all_features)
    
    # Normalized features
    normalized_2d = tsne.fit_transform(normalized_features)
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original features
    colors = ['red' if label == 0 else 'blue' for label in all_labels]
    ax1.scatter(original_2d[:, 0], original_2d[:, 1], c=colors, alpha=0.6)
    ax1.set_title('Original Features\n(Red: Child, Blue: Adult)')
    ax1.grid(True, alpha=0.3)
    
    # Normalized features
    ax2.scatter(normalized_2d[:, 0], normalized_2d[:, 1], c='green', alpha=0.6)
    ax2.set_title('Age-Normalized Features\n(All normalized to adult domain)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'age_normalization_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Saved t-SNE visualization: {save_dir / 'age_normalization_tsne.png'}")
    
    return normalized_features

# ==================== MAIN EXECUTION ====================
def main():
    parser = argparse.ArgumentParser(description='Phase 2: CycleGAN Age Normalization')
    parser.add_argument('--features-dir', required=True, help='Path to Phase 1 features')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--save-dir', default='phase2_results', help='Results save directory')
    parser.add_argument('--load-checkpoint', help='Path to checkpoint to resume training')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PHASE 2: CYCLEGAN AGE NORMALIZATION")
    print("=" * 60)
    
    # Load dataset
    print("[INFO] Loading feature dataset...")
    dataset = KinshipFeatureDataset(args.features_dir, feature_type='combined')
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                           drop_last=True, num_workers=4)
    
    # Initialize trainer
    trainer = CycleGANTrainer(feature_dim=512, learning_rate=args.learning_rate)
    
    # Load checkpoint if specified
    start_epoch = 0
    if args.load_checkpoint:
        start_epoch = trainer.load_checkpoint(Path(args.load_checkpoint))
    
    # Training
    trainer.train(dataloader, args.epochs, args.save_dir)
    
    # Evaluation
    print("\n[INFO] Evaluating age normalization...")
    normalized_features = evaluate_age_normalization(trainer, dataset, args.save_dir)
    
    # Save normalized features for Phase 3
    np.save(Path(args.save_dir) / 'normalized_features.npy', normalized_features)
    
    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETED")
    print("=" * 60)
    print(f"Results saved to: {args.save_dir}")
    print("Normalized features ready for Phase 3 (Siamese Network)")

if __name__ == '__main__':
    main()
