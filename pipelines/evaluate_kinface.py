"""
KinFaceW-II Dataset Evaluation Script

This script evaluates the kinship verification model on the KinFaceW-II dataset
by testing on random positive (kin) and negative (non-kin) pairs.

Usage:
    python evaluate_kinface.py --num-pairs 100 --output-dir results/

Requirements:
    - kinship_inference.py in the same directory
    - KinFaceW-II dataset in ../KinFaceW-II/images/
"""

import argparse
import sys
import os
from pathlib import Path
import subprocess
import random
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np

# Dataset path
DATASET_ROOT = Path(__file__).resolve().parent.parent / "KinFaceW-II" / "images"
INFERENCE_SCRIPT = Path(__file__).resolve().parent / "kinship_inference.py"

def get_pairs_from_category(category_path):
    """Get all positive pairs from a category (e.g., father-dau)"""
    pairs = []
    for img_file in category_path.iterdir():
        if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            # Extract pair ID from filename (e.g., fd_001_1.jpg -> 001)
            parts = img_file.stem.split('_')
            if len(parts) >= 3:
                pair_id = parts[1]
                person_id = parts[2]  # 1 or 2
                pairs.append((str(img_file), pair_id, person_id))
    # Group into pairs
    pair_dict = {}
    for img, pair_id, person_id in pairs:
        if pair_id not in pair_dict:
            pair_dict[pair_id] = []
        pair_dict[pair_id].append((img, person_id))
    positive_pairs = []
    for pair_id, imgs in pair_dict.items():
        if len(imgs) == 2:
            img1, img2 = sorted(imgs, key=lambda x: x[1])
            positive_pairs.append((img1[0], img2[0], 1))  # 1 for kin
    return positive_pairs

def get_all_positive_pairs():
    """Get all positive pairs from all categories"""
    categories = ['father-dau', 'father-son', 'mother-dau', 'mother-son']
    all_positives = []
    for cat in categories:
        cat_path = DATASET_ROOT / cat
        if cat_path.exists():
            pairs = get_pairs_from_category(cat_path)
            all_positives.extend(pairs)
    return all_positives

def generate_negative_pairs(all_positives, num_negatives):
    """Generate negative pairs (non-kin) randomly"""
    categories = ['father-dau', 'father-son', 'mother-dau', 'mother-son']
    all_images = []
    for cat in categories:
        cat_path = DATASET_ROOT / cat
        if cat_path.exists():
            for img_file in cat_path.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                    all_images.append(str(img_file))

    negatives = []
    used_pairs = set()
    while len(negatives) < num_negatives:
        img1 = random.choice(all_images)
        img2 = random.choice(all_images)
        if img1 != img2:
            pair_key = tuple(sorted([img1, img2]))
            if pair_key not in used_pairs:
                # Check if they are not from the same pair
                pair1 = img1.split('_')[1]
                pair2 = img2.split('_')[1]
                if pair1 != pair2:
                    negatives.append((img1, img2, 0))  # 0 for non-kin
                    used_pairs.add(pair_key)
    return negatives

def run_inference(image1, image2):
    """Run kinship_inference.py and get the prediction from JSON output"""
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            sys.executable, str(INFERENCE_SCRIPT),
            '--image1', image1,
            '--image2', image2,
            '--output-dir', temp_dir
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            # Check if the JSON file was created
            json_path = Path(temp_dir) / 'kinship_results.json'
            if json_path.exists():
                with open(json_path, 'r') as f:
                    data = json.load(f)
                return data['kinship_metrics']['is_kin'], data['kinship_metrics']['kinship_probability']
            else:
                # Fallback to parsing output
                output = result.stdout
                if "PREDICTION: KINSHIP DETECTED" in output:
                    return 1, None  # Kin
                elif "PREDICTION: NO KINSHIP DETECTED" in output:
                    return 0, None  # Non-kin
                else:
                    print(f"[WARN] Could not parse prediction from output for {image1} and {image2}")
                    print(f"[DEBUG] stdout: {output}")
                    print(f"[DEBUG] stderr: {result.stderr}")
                    return None, None
        except subprocess.TimeoutExpired:
            print(f"[ERROR] Inference timed out for {image1} and {image2}")
            return None, None
        except Exception as e:
            print(f"[ERROR] Error running inference: {e}")
            return None, None

def main():
    parser = argparse.ArgumentParser(description='Evaluate Kinship Verification on KinFaceW-II')
    parser.add_argument('--num-pairs', type=int, default=100, help='Number of pairs to test (half positive, half negative)')
    parser.add_argument('--output-dir', type=str, default='kinface_evaluation_results', help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("KINFACEW-II EVALUATION")
    print("=" * 60)

    # Get positive pairs
    positives = get_all_positive_pairs()
    print(f"[INFO] Found {len(positives)} positive pairs")

    # Generate negative pairs
    num_neg = args.num_pairs // 2
    negatives = generate_negative_pairs(positives, num_neg)
    print(f"[INFO] Generated {len(negatives)} negative pairs")

    # Combine and shuffle
    all_pairs = positives[:num_neg] + negatives  # Balance the dataset
    random.shuffle(all_pairs)

    print(f"[INFO] Testing on {len(all_pairs)} pairs")

    true_labels = []
    predictions = []
    scores = []

    for i, (img1, img2, label) in enumerate(all_pairs):
        print(f"\n[TEST {i+1}/{len(all_pairs)}] {img1} vs {img2}")
        result = run_inference(img1, img2)
        if result is not None and isinstance(result, tuple) and len(result) == 2:
            pred, score = result
            true_labels.append(label)
            predictions.append(pred)
            scores.append(score if score is not None else 0.5)  # Default score if None
            print(f"  True: {'Kin' if label else 'Non-Kin'}, Pred: {'Kin' if pred else 'Non-Kin'}, Score: {score:.2%}" if score is not None else f"  True: {'Kin' if label else 'Non-Kin'}, Pred: {'Kin' if pred else 'Non-Kin'}, Score: N/A")
        else:
            print(f"  [ERROR] Skipped pair")

    # Compute metrics
    if len(true_labels) > 0:
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        try:
            auc = roc_auc_score(true_labels, scores)
        except ValueError:
            auc = 0.0

        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Accuracy:  {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall:    {recall:.2%}")
        print(f"F1-Score:  {f1:.2%}")
        print(f"ROC-AUC:   {auc:.2%}")
        print("-" * 60)
        print(f"Confusion Matrix:")
        print(f"  TP: {tp}, FP: {fp}")
        print(f"  FN: {fn}, TN: {tn}")
        print("=" * 60)

        # Save results
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(auc),
            'confusion_matrix': {
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn)
            },
            'num_pairs': int(len(true_labels))
        }
        with open(output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[INFO] Results saved to {output_dir / 'evaluation_results.json'}")
    else:
        print("[ERROR] No pairs were evaluated successfully")

if __name__ == '__main__':
    main()
