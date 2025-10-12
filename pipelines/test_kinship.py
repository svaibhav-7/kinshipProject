# test_kinship.py
import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np

# 1. Siamese model definition (must match training)
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim=512, embedding_dim=128):
        super().__init__()
        self.embedding_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, embedding_dim)
        )
    def forward_once(self, x):
        return self.embedding_net(x)
    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

def load_model(ckpt_path, device):
    cp = torch.load(ckpt_path, map_location=device)
    cfg = cp['model_config']
    model = SiameseNetwork(cfg['input_dim'], cfg['embedding_dim']).to(device)
    model.load_state_dict(cp['model_state_dict'])
    model.eval()
    return model

def load_feature_mapping(metadata_json, features_npy):
    with open(metadata_json, 'r') as f:
        metadata = json.load(f)
    filenames = [entry['filename'] for entry in metadata]
    features = np.load(features_npy)
    # map image.jpg -> feature vector
    mapping = {}
    for idx, fname in enumerate(filenames):
        base = fname.replace('_combined_features.npy','')
        mapping[base + '.jpg'] = features[idx]
    return mapping

def infer(model, feat1, feat2, threshold, device):
    t1 = torch.tensor(feat1).float().unsqueeze(0).to(device)
    t2 = torch.tensor(feat2).float().unsqueeze(0).to(device)
    with torch.no_grad():
        e1, e2 = model(t1, t2)
        dist = nn.functional.pairwise_distance(e1, e2).item()
    kin = dist <= threshold
    degree = 1.0 / (1.0 + dist)
    return dist, kin, degree

def main(img1, img2):
    # Paths (update if needed)
    base = r"D:\SasiVaibhav\klu\3rd year\projects\kinship_verification\kinshipProject"
    ckpt = os.path.join(base, "siamese_kinship_all_relations.pt")
    metadata_json = os.path.join(base, "model_visuals_ganmodel", "features_with_metadata.json")
    features_npy = os.path.join(base, "model_visuals_ganmodel", "normalized_features.npy")
    # Optimized threshold (compute per-relation or use mean_threshold from evaluation_results.json)
    threshold = 0.504  # example value; replace with your calibrated threshold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt, device)
    mapping = load_feature_mapping(metadata_json, features_npy)

    if img1 not in mapping or img2 not in mapping:
        print(f"Error: features not found for {img1} or {img2}")
        sys.exit(1)

    feat1 = mapping[img1]
    feat2 = mapping[img2]
    dist, kin, degree = infer(model, feat1, feat2, threshold, device)

    print(f"Image 1: {img1}")
    print(f"Image 2: {img2}")
    print(f"Euclidean distance between embeddings: {dist:.4f}")
    print(f"Threshold for kin/non-kin: {threshold:.4f}")
    print(f"Predicted: {'Kin' if kin else 'NonKin'}")
    print(f"Degree of relation (1/(1+dist)): {degree:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_kinship.py <img1.jpg> <img2.jpg>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
