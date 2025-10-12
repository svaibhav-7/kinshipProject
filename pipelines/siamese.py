import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import scipy.io
import json
import os

# -- Dataset for multiple kinship relations --
class KinFacePairsDataset(Dataset):
    def __init__(self, mat_file_path, normalized_features, metadata_filenames, relation_type):
        self.relation_type = relation_type
        self.pairs_data = scipy.io.loadmat(mat_file_path)
        
        print(f"Loading {relation_type} pairs from {os.path.basename(mat_file_path)}")
        
        # The actual structure: pairs contains [fold, kin/non-kin, image1, image2]
        pairs = self.pairs_data['pairs']
        print(f"Pairs shape: {pairs.shape}")
        
        # Extract data from the pairs array
        # Assuming pairs has shape (N, 4) where columns are [fold, kin/non-kin, image1, image2]
        self.fold_array = pairs[:, 0]  # fold number
        self.kin_array = pairs[:, 1]   # kin/non-kin (1 for kin, 0 for non-kin)
        self.img1_array = [str(pairs[i, 2][0]) if hasattr(pairs[i, 2], '__getitem__') else str(pairs[i, 2]) for i in range(len(pairs))]
        self.img2_array = [str(pairs[i, 3][0]) if hasattr(pairs[i, 3], '__getitem__') else str(pairs[i, 3]) for i in range(len(pairs))]
        
        self.normalized_features = normalized_features
        self.filename_to_index = {fname: idx for idx, fname in enumerate(metadata_filenames)}
        
        print(f"{relation_type} dataset loaded: {len(self.kin_array)} pairs")
        print(f"Sample data - Fold: {self.fold_array[0]}, Label: {self.kin_array[0]}")
        print(f"Sample images: {self.img1_array[0]}, {self.img2_array[0]}")

    def __len__(self):
        return len(self.kin_array)

    def __getitem__(self, idx):
        label = int(self.kin_array[idx])
        fname1 = self.img1_array[idx]
        fname2 = self.img2_array[idx]

        # Convert image filenames to feature filenames if needed
        # Check if filename already has extension, if not add .jpg
        if not fname1.endswith(('.jpg', '.png', '.npy')):
            fname1 += '.jpg'
        if not fname2.endswith(('.jpg', '.png', '.npy')):
            fname2 += '.jpg'
            
        fname1_feature = fname1.replace('.jpg', '_combined_features.npy').replace('.png', '_combined_features.npy')
        fname2_feature = fname2.replace('.jpg', '_combined_features.npy').replace('.png', '_combined_features.npy')

        idx1 = self.filename_to_index.get(fname1_feature, self.filename_to_index.get(fname1))
        idx2 = self.filename_to_index.get(fname2_feature, self.filename_to_index.get(fname2))

        if idx1 is None or idx2 is None:
            print(f"Missing features for: {fname1} -> {fname1_feature}, {fname2} -> {fname2_feature}")
            print(f"Available filenames sample: {list(self.filename_to_index.keys())[:3]}")
            raise ValueError(f"Feature index missing for {fname1} or {fname2}")

        feat1 = torch.tensor(self.normalized_features[idx1], dtype=torch.float32)
        feat2 = torch.tensor(self.normalized_features[idx2], dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return (feat1, feat2), label

# -- Alternative: Debug dataset to understand structure better --
def debug_mat_file(mat_file_path):
    """Debug function to understand the exact structure of mat files"""
    pairs_data = scipy.io.loadmat(mat_file_path)
    print(f"\nDebugging {os.path.basename(mat_file_path)}:")
    print("Keys:", list(pairs_data.keys()))
    
    pairs = pairs_data['pairs']
    print(f"Pairs shape: {pairs.shape}")
    print(f"Pairs dtype: {pairs.dtype}")
    
    # Look at first few entries
    for i in range(min(3, len(pairs))):
        print(f"Entry {i}: {pairs[i]}")
        for j in range(pairs.shape[1]):
            print(f"  Column {j}: {pairs[i, j]} (type: {type(pairs[i, j])})")

# -- Siamese Model and Loss (unchanged) --
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim=512, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        self.embedding_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, embedding_dim)
        )
    def forward_once(self, x):
        return self.embedding_net(x)
    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss

# -- Training Loop with All Relations --
def train_siamese_all_relations():
    # Base paths
    base_path = r"D:\SasiVaibhav\klu\3rd year\projects\kinship_verification\kinshipProject"
    meta_data_path = os.path.join(base_path, "KinFaceW-II", "meta_data")
    features_path = os.path.join(base_path, "model_visuals_ganmodel", "normalized_features.npy")
    metadata_json_path = os.path.join(base_path, "model_visuals_ganmodel", "features_with_metadata.json")

    # All kinship relation files
    relation_files = {
        "father-daughter": "fd_pairs.mat",
        "father-son": "fs_pairs.mat", 
        "mother-daughter": "md_pairs.mat",
        "mother-son": "ms_pairs.mat"
    }

    # First, debug the mat files to understand structure
    for relation_type, mat_file in relation_files.items():
        mat_file_path = os.path.join(meta_data_path, mat_file)
        if os.path.exists(mat_file_path):
            debug_mat_file(mat_file_path)
            break  # Just debug one file to understand structure

    # Load features and metadata
    normalized_features = np.load(features_path)
    with open(metadata_json_path, "r") as f:
        feature_metadata = json.load(f)
    metadata_filenames = [entry['filename'] for entry in feature_metadata]

    print(f"\nLoaded {len(normalized_features)} features and {len(metadata_filenames)} filenames")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets for all relations
    datasets = []
    for relation_type, mat_file in relation_files.items():
        mat_file_path = os.path.join(meta_data_path, mat_file)
        
        if os.path.exists(mat_file_path):
            try:
                dataset = KinFacePairsDataset(mat_file_path, normalized_features, 
                                            metadata_filenames, relation_type)
                datasets.append(dataset)
            except Exception as e:
                print(f"Error loading {relation_type}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"File not found: {mat_file_path}")

    if not datasets:
        print("No datasets loaded successfully!")
        return

    # Combine all datasets
    combined_dataset = ConcatDataset(datasets)
    print(f"Combined dataset size: {len(combined_dataset)} pairs")
    
    dataloader = DataLoader(combined_dataset, batch_size=32, shuffle=True, num_workers=0)

    # Initialize model
    model = SiameseNetwork(input_dim=normalized_features.shape[1]).to(device)
    criterion = ContrastiveLoss(margin=2.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    model.train()
    for epoch in range(200):
        total_loss = 0.0
        num_batches = 0
        for (feat1, feat2), labels in dataloader:
            feat1, feat2, labels = feat1.to(device), feat2.to(device), labels.to(device)
            out1, out2 = model(feat1, feat2)
            loss = criterion(out1, out2, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch + 1}/200], Loss: {avg_loss:.4f}")
    
    print("Training complete!")
    
    
    # Save the complete model as .pt
    model_save_path = os.path.join(base_path, 'siamese_kinship_all_relations.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': normalized_features.shape[1],
            'embedding_dim': 128
        },
        'training_info': {
            'epochs': 20,
            'final_loss': avg_loss,
            'num_pairs': len(combined_dataset)
        }
    }, model_save_path)

    print(f"Model saved as '{model_save_path}'")
    print(f"Model saved as '{model_save_path}'")
    
    return model

if __name__ == "__main__":
    train_siamese_all_relations()
