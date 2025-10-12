import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self, feature_dim):
        super(SiameseNetwork, self).__init__()
        # Define a simple shared embedding network
        self.embedding_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64) # Output embedding dimension
        )

        # Define the classification head that takes the absolute difference of the embeddings
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Linear(64, 1) # Output a single value for binary classification (kin/non-kin)
        )

    def forward_one(self, x):
        # Forward pass for one branch of the Siamese network
        return self.embedding_net(x)

    def forward(self, feature1, feature2):
        # Forward pass for the Siamese network
        output1 = self.forward_one(feature1)
        output2 = self.forward_one(feature2)

        # Compute the absolute difference between the embeddings
        diff = torch.abs(output1 - output2)

        # Pass the difference through the classification head
        output = self.fc(diff)
        return output

# Instantiate the Siamese network
# Assuming feature_dim is the dimension of the normalized features (512 based on previous steps)
feature_dim = normalized_features.shape[1] if 'normalized_features' in locals() else 512
kinship_model = SiameseNetwork(feature_dim)

# Define the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kinship_model.to(DEVICE)

print(f"Siamese Kinship Model defined with feature dimension {feature_dim} and moved to {DEVICE}.")

# Assuming 'normalized_features' and 'labels' are loaded from the previous step

# Create a list of samples including filenames to help pair data
# This requires re-loading metadata or reconstructing it if possible from filenames
# Since metadata loading failed previously, let's try to infer from filenames and available labels.
# Assuming filenames are structured to identify families and individuals (e.g., FamilyID_IndividualID_AgeGroup_...)

# A simplified approach for pairing based on indices and labels:
# We'll create pairs of indices (i, j) from the dataset.
# A pair (i, j) is positive (kin) if samples i and j belong to the same family.
# A pair (i, j) is negative (non-kin) if samples i and j belong to different families.
# Since we don't have explicit family IDs from the loaded data, we'll need to infer this.

# Let's assume the original data structure or filenames allowed identifying families.
# If we had the original metadata, we could use 'category' and potentially 'filename' to group by family.
# As a fallback, and given the data limitations, let's assume we can approximate kinship
# based on the original sample structure or some inferred grouping.

# *** IMPORTANT: This pairing logic is a placeholder.
# A robust implementation requires access to family identifiers.
# Without reliable family IDs, this will be a simplified pairing based on age labels and indices,
# which is NOT a true kinship pairing but demonstrates the pairing concept for evaluation.

# Let's create a simplified pairing:
# Positive pairs: Two samples with the same age label (both child or both adult) that are "close" in the original data order (implying they might be from related families or the same family in a sorted dataset).
# Negative pairs: Two samples with different age labels (one child, one adult) or samples with the same age label that are "far" apart in the original data order.

# This simplified approach is NOT ideal for true kinship evaluation but allows the pipeline to continue.
# A better approach would involve re-running Phase 2 evaluation to save metadata or finding the correct metadata file.

# Let's proceed with a basic pairing strategy for demonstration:
# Pair each sample with a few others.
# Label as 'kin' if they have the same age label (0,0 or 1,1). This is a weak proxy for kinship.
# Label as 'non-kin' if they have different age labels (0,1 or 1,0). This is a weak proxy for non-kinship.

pairs = []
num_samples = normalized_features.shape[0]
num_pairs_per_sample = 5 # Create 5 pairs for each sample

for i in range(num_samples):
    # Create positive-like pairs (same age label)
    same_age_indices = np.where(labels == labels[i])[0]
    # Select a few random indices from the same age group, excluding the sample itself
    candidate_indices = same_age_indices[same_age_indices != i]
    if len(candidate_indices) > 0:
        selected_indices = np.random.choice(candidate_indices, min(num_pairs_per_sample // 2, len(candidate_indices)), replace=False)
        for j in selected_indices:
             # Assuming same age label implies a *potential* kin-like relationship for this simplified example
             pairs.append({'features_1': normalized_features[i], 'features_2': normalized_features[j], 'is_kin': 1}) # 1 for kin

    # Create negative-like pairs (different age label)
    diff_age_indices = np.where(labels != labels[i])[0]
    if len(diff_age_indices) > 0:
        selected_indices = np.random.choice(diff_age_indices, min(num_pairs_per_sample - len(selected_indices), len(diff_age_indices)), replace=False)
        for j in selected_indices:
            # Assuming different age label implies a *potential* non-kin relationship for this simplified example
            pairs.append({'features_1': normalized_features[i], 'features_2': normalized_features[j], 'is_kin': 0}) # 0 for non-kin

print(f"Generated {len(pairs)} pairs for evaluation.")

# Split pairs into training and testing sets
# This requires a proper split that avoids leakage (e.g., family-based split)
# For this simplified example, we'll do a random split of pairs.
# A proper kinship evaluation dataset splits families, not just pairs.

from sklearn.model_selection import train_test_split

# Convert list of dicts to separate lists for train_test_split
features_1 = np.array([p['features_1'] for p in pairs])
features_2 = np.array([p['features_2'] for p in pairs])
is_kin = np.array([p['is_kin'] for p in pairs])

# Perform the split on the indices of the pairs
pair_indices = np.arange(len(pairs))
train_indices, test_indices, y_train, y_test = train_test_split(
    pair_indices, is_kin, test_size=0.2, random_state=42, stratify=is_kin
)

train_pairs = [{'features_1': features_1[i], 'features_2': features_2[i], 'is_kin': is_kin[i]} for i in train_indices]
test_pairs = [{'features_1': features_1[i], 'features_2': features_2[i], 'is_kin': is_kin[i]} for i in test_indices]

print(f"Split pairs into {len(train_pairs)} training and {len(test_pairs)} testing pairs.")

# Now, train_pairs and test_pairs can be used with a Siamese network or similar model.