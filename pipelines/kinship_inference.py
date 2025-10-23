"""
Kinship Verification Inference Script

This script takes two images from user, extracts features, normalizes them,
visualizes the normalized faces, and evaluates kinship using trained models.

Usage:
    python kinship_inference.py --image1 path/to/image1.jpg --image2 path/to/image2.jpg
    
Models required:
    - final_model.pt: GAN normalization model
    - decoder_model.pth: Feature-to-image decoder
    - siamese_kinship_all_relations.pt: Siamese kinship classifier
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import timm

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent
KINSHIP_PROJECT_ROOT = PROJECT_ROOT.parent

# Import Facemesh feature extractor
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from Facemesh import EnhancedHybridFeatureExtractor
except ImportError:
    print("[WARN] Could not import EnhancedHybridFeatureExtractor. Using fallback.")
    EnhancedHybridFeatureExtractor = None

# Import decoder architecture
try:
    from decoder_arch import ImprovedFeatureToImageDecoder
except ImportError:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from decoder_arch import ImprovedFeatureToImageDecoder

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {DEVICE}")

# ==================== FACE DETECTION AND PREPROCESSING ====================

def detect_and_align_face(image_path, img_size=224):
    """
    Detect face in the image and align it for better feature extraction.
    
    Args:
        image_path: Path to the input image
        img_size: Desired size for the output face image
    
    Returns:
        aligned_face: Cropped and aligned face image (numpy array)
        face_detected: Boolean indicating if a face was detected
    """
    print(f"[INFO] Detecting and aligning face in: {image_path}")
    
    # Load the image
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces with relaxed parameters for KinFaceW-II
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=1, minSize=(10, 10))
    
    if len(faces) == 0:
        # Try even more relaxed
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(5, 5))
    
    if len(faces) == 0:
        print(f"[WARN] No face detected in {image_path}. Using center crop as fallback.")
        # Fallback: use center crop
        h, w = img.shape[:2]
        size = min(h, w)
        start_x = (w - size) // 2
        start_y = (h - size) // 2
        face_crop = img[start_y:start_y+size, start_x:start_x+size]
        aligned_face = cv2.resize(face_crop, (img_size, img_size))
        return aligned_face, False
    
    # Use the largest detected face
    if len(faces) > 1:
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    
    (x, y, w, h) = faces[0]
    
    # Add padding for better alignment (30% expansion)
    expansion = 0.3
    expand_x = int(w * expansion)
    expand_y = int(h * expansion)
    
    x = max(0, x - expand_x)
    y = max(0, y - expand_y)
    w = min(img.shape[1] - x, w + 2 * expand_x)
    h = min(img.shape[0] - y, h + 2 * expand_y)
    
    # Crop the face
    face = img[y:y+h, x:x+w]
    
    # Resize to desired size
    aligned_face = cv2.resize(face, (img_size, img_size))
    
    print(f"[INFO] Face detected and aligned: {aligned_face.shape}")
    return aligned_face, True

# ==================== MODEL ARCHITECTURES ====================

class FeatureGenerator(nn.Module):
    """Generator network for feature-space normalization (from GAN_norm.py)"""
    
    def __init__(self, feature_dim=512, hidden_dims=[256, 128, 256]):
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
            nn.Tanh()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        encoded = self.encoder(x)
        bottleneck = self.bottleneck(encoded)
        decoded = self.decoder(bottleneck)
        output = decoded + self.residual_weight * x
        return output

class SiameseNetwork(nn.Module):
    """Siamese network for kinship verification with L2 normalization"""
    
    def __init__(self, input_dim=512, embedding_dim=128):
        super().__init__()
        self.embedding_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )
    
    def forward_once(self, x):
        embedding = self.embedding_net(x)
        # CRITICAL: L2 normalize embeddings to unit sphere
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding
    
    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2

# ==================== FEATURE EXTRACTION ====================

def extract_features_from_image(img_path, feature_extractor, img_size=224):
    """
    Extract features using the SAME method as training (Facemesh.py) with face detection
    
    Args:
        img_path: Path to input image
        feature_extractor: Feature extractor instance
        img_size: Image size for preprocessing
    
    Returns:
        combined_features: Feature vector (matches training)
    """
    print(f"[INFO] Extracting features from: {img_path}")
    
    # Detect and align face
    aligned_face, face_detected = detect_and_align_face(img_path, img_size)
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
    
    # Create synthetic depth map as in training (from Facemesh.py feature-only mode)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Method 1: Edge-based depth
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2).astype(np.float32)
    edge_depth = 1.0 - (gradient_magnitude / (gradient_magnitude.max() + 1e-8))
    
    # Method 2: Laplacian-based depth
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    laplacian_depth = 1.0 - np.abs(laplacian) / (np.abs(laplacian).max() + 1e-8)
    
    # Method 3: Gaussian blur-based depth
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    curvature_depth = blurred.astype(np.float32) / 255.0
    
    # Combine methods
    depth_canvas = (
        0.4 * edge_depth +           # Edge information
        0.3 * laplacian_depth +      # Structural information  
        0.3 * curvature_depth        # Curvature information
    )
    
    # Apply bilateral filter for smoother depth
    depth_canvas = cv2.bilateralFilter(depth_canvas.astype(np.float32), 9, 75, 75)
    
    # Convert to tensors
    rgb_tensor = torch.from_numpy(img_rgb.astype(np.float32)).permute(2, 0, 1) / 255.0
    depth_tensor = torch.from_numpy(depth_canvas.astype(np.float32)).unsqueeze(0)
    
    # Extract features using Facemesh method (matches training!)
    combined_features, _ = feature_extractor.extract_combined_features(
        rgb_tensor, depth_tensor, use_fusion=True  # 512 dims to match GAN model
    )
    
    print(f"[INFO] Extracted feature vector of shape: {combined_features.shape}")
    if not face_detected:
        print(f"[WARN] No face detected, used center crop. Results may be less accurate.")
    else:
        print(f"[INFO] Face detected successfully, using aligned face.")
    
    return combined_features

# ==================== FEATURE NORMALIZATION ====================

def normalize_features(features):
    """L2 normalize feature vectors"""
    norm = np.linalg.norm(features)
    if norm > 1e-8:
        return features / norm
    return features

# ==================== VISUALIZATION ====================

def denorm(t):
    """Map [-1, 1] -> [0, 1] and clamp"""
    return ((t + 1) / 2).clamp(0, 1)

def visualize_normalized_faces(original_img1, original_img2, 
                                normalized_feat1, normalized_feat2, 
                                decoder_model, save_path):
    """
    Visualize original and normalized (decoded) faces
    
    Args:
        original_img1, original_img2: Original images (PIL or numpy)
        normalized_feat1, normalized_feat2: Normalized feature vectors
        decoder_model: Decoder model to generate images from features
        save_path: Path to save visualization
    """
    print("[INFO] Generating face visualizations...")
    
    # Convert features to tensors
    feat1_tensor = torch.from_numpy(normalized_feat1).float().unsqueeze(0).to(DEVICE)
    feat2_tensor = torch.from_numpy(normalized_feat2).float().unsqueeze(0).to(DEVICE)
    
    # Generate images from features
    with torch.no_grad():
        gen_img1 = decoder_model(feat1_tensor)
        gen_img2 = decoder_model(feat2_tensor)
    
    # Denormalize
    gen_img1 = denorm(gen_img1).cpu().squeeze(0).permute(1, 2, 0).numpy()
    gen_img2 = denorm(gen_img2).cpu().squeeze(0).permute(1, 2, 0).numpy()
    
    # Convert original images to numpy if needed
    if isinstance(original_img1, str) or isinstance(original_img1, Path):
        original_img1 = np.array(Image.open(original_img1).convert('RGB'))
    if isinstance(original_img2, str) or isinstance(original_img2, Path):
        original_img2 = np.array(Image.open(original_img2).convert('RGB'))
    
    # Resize originals to match generated size
    gen_size = gen_img1.shape[:2]
    orig_img1_resized = cv2.resize(original_img1, (gen_size[1], gen_size[0]))
    orig_img2_resized = cv2.resize(original_img2, (gen_size[1], gen_size[0]))
    
    # Normalize to [0, 1]
    orig_img1_resized = orig_img1_resized.astype(np.float32) / 255.0
    orig_img2_resized = orig_img2_resized.astype(np.float32) / 255.0
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[0, 0].imshow(orig_img1_resized)
    axes[0, 0].set_title("Original Image 1", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gen_img1)
    axes[0, 1].set_title("Normalized Image 1", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(orig_img2_resized)
    axes[1, 0].set_title("Original Image 2", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(gen_img2)
    axes[1, 1].set_title("Normalized Image 2", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Visualization saved to: {save_path}")

# ==================== KINSHIP EVALUATION ====================

def evaluate_kinship(feat1, feat2, siamese_model, threshold=1.0):
    """
    Evaluate kinship probability between two feature vectors
    
    Args:
        feat1, feat2: Feature vectors (numpy arrays) - should be normalized
        siamese_model: Trained Siamese network
        threshold: Decision threshold (default 1.0 for normalized embeddings)
    
    Returns:
        result: Dictionary with kinship prediction and metrics
    """
    print("[INFO] Evaluating kinship...")
    
    # Convert to tensors
    feat1_tensor = torch.from_numpy(feat1).float().unsqueeze(0).to(DEVICE)
    feat2_tensor = torch.from_numpy(feat2).float().unsqueeze(0).to(DEVICE)
    
    # Get embeddings (now L2 normalized inside the model)
    with torch.no_grad():
        emb1, emb2 = siamese_model(feat1_tensor, feat2_tensor)
        
        # Compute Euclidean distance on normalized embeddings
        distance = F.pairwise_distance(emb1, emb2)
        distance_value = distance.item()
        
        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(emb1, emb2).item()
        
        # Compute kinship probability (distance closer to 0 = higher probability)
        # For normalized embeddings, distance is in [0, 2]
        kinship_probability = 1.0 / (1.0 + distance_value)
        
        # DEBUG: Print raw values
        print(f"[DEBUG] Euclidean Distance: {distance_value:.4f}")
        print(f"[DEBUG] Cosine Similarity: {cosine_sim:.4f}")
        print(f"[DEBUG] Threshold: {threshold:.4f}")
        print(f"[DEBUG] Embedding 1 norm: {torch.norm(emb1).item():.4f} (should be ~1.0)")
        print(f"[DEBUG] Embedding 2 norm: {torch.norm(emb2).item():.4f} (should be ~1.0)")
        
        # Make prediction based on distance (lower distance = kin)
        is_kin = distance_value <= threshold
        
        # Confidence score
        if is_kin:
            confidence = (threshold - distance_value) / threshold
        else:
            confidence = (distance_value - threshold) / 2.0
    
    result = {
        'euclidean_distance': distance_value,
        'cosine_similarity': cosine_sim,
        'kinship_probability': kinship_probability,
        'is_kin': is_kin,
        'threshold': threshold,
        'confidence': confidence
    }
    
    return result

# ==================== MAIN PIPELINE ====================

def main():
    parser = argparse.ArgumentParser(description='Kinship Verification from Two Images')
    parser.add_argument('--image1', type=str, required=True, help='Path to first image')
    parser.add_argument('--image2', type=str, required=True, help='Path to second image')
    parser.add_argument('--gan-model', type=str, 
                        default=str(KINSHIP_PROJECT_ROOT / 'model_visuals_ganmodel' / 'final_model.pt'),
                        help='Path to GAN normalization model')
    parser.add_argument('--decoder-model', type=str,
                        default=str(KINSHIP_PROJECT_ROOT / 'decoder_model' / 'decoder_model.pth'),
                        help='Path to decoder model')
    parser.add_argument('--siamese-model', type=str,
                        default=str(KINSHIP_PROJECT_ROOT / 'final_siamese_model' / 'siamese_kinship_all_relations.pt'),
                        help='Path to Siamese kinship model')
    parser.add_argument('--output-dir', type=str,
                        default=str(KINSHIP_PROJECT_ROOT / 'kinship_inference_results'),
                        help='Output directory for results')
    parser.add_argument('--img-size', type=int, default=224, help='Input image size')
    parser.add_argument('--threshold', type=float, default=0.3, 
                        help='Kinship decision threshold (default: 0.3 for normalized embeddings)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("KINSHIP VERIFICATION INFERENCE PIPELINE")
    print("=" * 80)
    print(f"Image 1: {args.image1}")
    print(f"Image 2: {args.image2}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Verify input images exist
    img1_path = Path(args.image1)
    img2_path = Path(args.image2)
    
    if not img1_path.exists():
        raise FileNotFoundError(f"Image 1 not found: {img1_path}")
    if not img2_path.exists():
        raise FileNotFoundError(f"Image 2 not found: {img2_path}")
    
    # ===== STEP 1: Initialize Feature Extractor =====
    print("\n[STEP 1] Initializing feature extractor...")
    
    if EnhancedHybridFeatureExtractor is None:
        raise ImportError("EnhancedHybridFeatureExtractor not available. Check Facemesh.py")
    
    feature_extractor = EnhancedHybridFeatureExtractor(device=DEVICE)
    print("[INFO] Feature extractor initialized (using Facemesh method)")
    
    # ===== STEP 2: Extract Features =====
    print("\n[STEP 2] Extracting features from images...")
    
    features1 = extract_features_from_image(img1_path, feature_extractor, args.img_size)
    features2 = extract_features_from_image(img2_path, feature_extractor, args.img_size)
    
    # Compute raw feature distance
    raw_distance = np.linalg.norm(features1 - features2)
    print(f"[INFO] Raw feature distance before normalization: {raw_distance:.4f}")
    
    # Save raw features
    np.save(output_dir / 'features_image1.npy', features1)
    np.save(output_dir / 'features_image2.npy', features2)
    print(f"[INFO] Raw features saved to {output_dir}")
    
    # ===== STEP 2.5: Normalize Input Features =====
    print("\n[STEP 2.5] Normalizing input features...")
    features1_norm = np.linalg.norm(features1)
    features2_norm = np.linalg.norm(features2)
    features1 = normalize_features(features1)
    features2 = normalize_features(features2)
    print(f"[INFO] Features normalized (original norms: {features1_norm:.2f}, {features2_norm:.2f})")
    print(f"[INFO] Normalized feature norms: {np.linalg.norm(features1):.4f}, {np.linalg.norm(features2):.4f}")
    
    # ===== STEP 3: Load GAN Normalization Model =====
    print("\n[STEP 3] Loading GAN normalization model...")
    
    gan_model_path = Path(args.gan_model)
    if not gan_model_path.exists():
        raise FileNotFoundError(f"GAN model not found: {gan_model_path}")
    
    # Load GAN model checkpoint
    gan_checkpoint = torch.load(gan_model_path, map_location=DEVICE, weights_only=False)
    
    # Initialize GAN generator
    feature_dim = features1.shape[0]
    gan_generator = FeatureGenerator(feature_dim=feature_dim).to(DEVICE)
    
    # Load state dict (handle different checkpoint formats)
    if 'G_child_to_adult' in gan_checkpoint:
        gan_generator.load_state_dict(gan_checkpoint['G_child_to_adult'], strict=False)
    elif 'G_c2a' in gan_checkpoint:
        gan_generator.load_state_dict(gan_checkpoint['G_c2a'], strict=False)
    elif 'model_state_dict' in gan_checkpoint:
        gan_generator.load_state_dict(gan_checkpoint['model_state_dict'], strict=False)
    else:
        gan_generator.load_state_dict(gan_checkpoint, strict=False)
    
    gan_generator.eval()
    print("[INFO] GAN normalization model loaded")
    
    # ===== STEP 4: Normalize Features with GAN =====
    print("\n[STEP 4] Normalizing features with GAN...")
    
    with torch.no_grad():
        feat1_tensor = torch.from_numpy(features1).float().unsqueeze(0).to(DEVICE)
        feat2_tensor = torch.from_numpy(features2).float().unsqueeze(0).to(DEVICE)
        
        normalized_feat1 = gan_generator(feat1_tensor).cpu().squeeze(0).numpy()
        normalized_feat2 = gan_generator(feat2_tensor).cpu().squeeze(0).numpy()
    
    # ===== STEP 4.5: Re-normalize After GAN =====
    print("\n[STEP 4.5] Re-normalizing features after GAN...")
    normalized_feat1 = normalize_features(normalized_feat1)
    normalized_feat2 = normalize_features(normalized_feat2)
    print(f"[INFO] Normalized feature norms: {np.linalg.norm(normalized_feat1):.4f}, {np.linalg.norm(normalized_feat2):.4f}")
    
    # Save normalized features
    np.save(output_dir / 'normalized_features_image1.npy', normalized_feat1)
    np.save(output_dir / 'normalized_features_image2.npy', normalized_feat2)
    print(f"[INFO] Normalized features saved to {output_dir}")
    
    # ===== STEP 5: Load Decoder Model =====
    print("\n[STEP 5] Loading decoder model...")
    
    decoder_model_path = Path(args.decoder_model)
    if not decoder_model_path.exists():
        raise FileNotFoundError(f"Decoder model not found: {decoder_model_path}")
    
    decoder_model = ImprovedFeatureToImageDecoder(feature_dim=feature_dim, img_size=64).to(DEVICE)
    
    # Load decoder checkpoint
    decoder_checkpoint = torch.load(decoder_model_path, map_location=DEVICE, weights_only=False)
    if 'generator' in decoder_checkpoint:
        decoder_model.load_state_dict(decoder_checkpoint['generator'], strict=False)
    else:
        decoder_model.load_state_dict(decoder_checkpoint, strict=False)
    
    decoder_model.eval()
    print("[INFO] Decoder model loaded")
    
    # ===== STEP 6: Visualize Normalized Faces =====
    print("\n[STEP 6] Generating face visualizations...")
    
    vis_path = output_dir / 'normalized_faces_comparison.png'
    visualize_normalized_faces(
        img1_path, img2_path,
        normalized_feat1, normalized_feat2,
        decoder_model, vis_path
    )
    
    # ===== STEP 7: Load Siamese Model =====
    print("\n[STEP 7] Loading Siamese kinship model...")
    
    siamese_model_path = Path(args.siamese_model)
    if not siamese_model_path.exists():
        raise FileNotFoundError(f"Siamese model not found: {siamese_model_path}")
    
    siamese_checkpoint = torch.load(siamese_model_path, map_location=DEVICE, weights_only=False)
    
    # Get model configuration
    if 'model_config' in siamese_checkpoint:
        input_dim = siamese_checkpoint['model_config'].get('input_dim', feature_dim)
        embedding_dim = siamese_checkpoint['model_config'].get('embedding_dim', 128)
    else:
        input_dim = feature_dim
        embedding_dim = 128
    
    siamese_model = SiameseNetwork(input_dim=input_dim, embedding_dim=embedding_dim).to(DEVICE)
    
    # Load state dict
    if 'model_state_dict' in siamese_checkpoint:
        siamese_model.load_state_dict(siamese_checkpoint['model_state_dict'], strict=False)
    else:
        siamese_model.load_state_dict(siamese_checkpoint, strict=False)
    
    siamese_model.eval()
    print("[INFO] Siamese kinship model loaded")
    
    # ===== STEP 8: Evaluate Kinship =====
    print("\n[STEP 8] Evaluating kinship...")
    
    kinship_result = evaluate_kinship(
        normalized_feat1, normalized_feat2,
        siamese_model, threshold=args.threshold
    )
    
    # ===== STEP 9: Display Results =====
    print("\n" + "=" * 80)
    print("KINSHIP VERIFICATION RESULTS")
    print("=" * 80)
    print(f"Euclidean Distance:      {kinship_result['euclidean_distance']:.4f}")
    print(f"Cosine Similarity:       {kinship_result['cosine_similarity']:.4f}")
    print(f"Kinship Probability:     {kinship_result['kinship_probability']:.2%}")
    print(f"Decision Threshold:      {kinship_result['threshold']:.4f}")
    print(f"Confidence:              {kinship_result['confidence']:.4f}")
    print("-" * 80)
    
    if kinship_result['is_kin']:
        print("PREDICTION: KINSHIP DETECTED (Related)")
    else:
        print("PREDICTION: NO KINSHIP DETECTED (Not Related)")
    
    print("=" * 80)
    
    # Save results to JSON
    import json
    results_json_path = output_dir / 'kinship_results.json'
    with open(results_json_path, 'w') as f:
        json.dump({
            'image1': str(img1_path),
            'image2': str(img2_path),
            'kinship_metrics': {
                'euclidean_distance': float(kinship_result['euclidean_distance']),
                'cosine_similarity': float(kinship_result['cosine_similarity']),
                'kinship_probability': float(kinship_result['kinship_probability']),
                'is_kin': bool(kinship_result['is_kin']),
                'threshold': float(kinship_result['threshold']),
                'confidence': float(kinship_result['confidence'])
            },
            'visualization': str(vis_path)
        }, f, indent=2)
    
    print(f"\n[INFO] Results saved to: {results_json_path}")
    print(f"[INFO] Visualization saved to: {vis_path}")
    print("\nInference complete!")

if __name__ == '__main__':
    main()
