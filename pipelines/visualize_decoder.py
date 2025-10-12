import argparse
from pathlib import Path
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os

# Import decoder architecture from local module (side-effect free)
try:
    from .decoder_arch import ImprovedFeatureToImageDecoder
except Exception:
    this_dir = Path(__file__).resolve().parent
    if str(this_dir) not in sys.path:
        sys.path.insert(0, str(this_dir))
    from decoder_arch import ImprovedFeatureToImageDecoder


def denorm(t: torch.Tensor) -> torch.Tensor:
    return ((t + 1) / 2).clamp(0, 1)


def visualize_pairs_grid(originals: torch.Tensor, generated: torch.Tensor, save_path: Path, cols: int = 5):
    n = min(originals.size(0), generated.size(0))
    cols = min(cols, n)
    rows = 2
    plt.figure(figsize=(cols * 3, rows * 3))
    for j in range(cols):
        # Original (top row)
        plt.subplot(rows, cols, j + 1)
        plt.imshow(originals[j].permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.title('Original')
        # Generated (bottom row)
        plt.subplot(rows, cols, cols + j + 1)
        plt.imshow(generated[j].permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.title('Generated (Normalized)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    try:
        plt.show()
    except Exception:
        pass
    plt.close()


def collect_images(images_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(Path(root) / f)
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description='Visualize decoder: originals (top) vs generated (bottom)')
    parser.add_argument('--features-path', type=str, default=str(Path('model_visuals_ganmodel') / 'normalized_features.npy'))
    parser.add_argument('--images-dir', type=str, default=str(Path('KinFaceW-II') / 'images'))
    parser.add_argument('--checkpoint', type=str, default=str(Path(__file__).resolve().parents[1] / 'decoder_model' / 'decoder_model.pth'))
    parser.add_argument('--save-dir', type=str, default=str(Path('model_visuals_ganmodel') / 'visualizations'))
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--img-size', type=int, default=64)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--pairs-grid-cols', type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # Resolve project root and paths
    project_root = Path(__file__).resolve().parents[1]
    features_path = Path(args.features_path)
    images_dir = Path(args.images_dir)
    checkpoint = Path(args.checkpoint)
    save_dir = Path(args.save_dir)
    if not features_path.is_absolute():
        features_path = project_root / features_path
    if not images_dir.is_absolute():
        images_dir = project_root / images_dir
    if not checkpoint.is_absolute():
        checkpoint = project_root / checkpoint
    if not save_dir.is_absolute():
        save_dir = project_root / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    if not features_path.exists():
        raise FileNotFoundError(f'Features not found: {features_path}')
    features = np.load(str(features_path))
    if features.ndim != 2:
        raise ValueError(f'Expected features [N,D], got {features.shape}')

    # Collect images
    if not images_dir.exists():
        raise FileNotFoundError(f'Images dir not found: {images_dir}')
    image_files = collect_images(images_dir)
    if len(image_files) == 0:
        raise RuntimeError(f'No images found in {images_dir}')

    # Choose first N aligned samples
    n = min(args.num_samples, len(features), len(image_files))
    feats = torch.from_numpy(features[:n]).float().to(device)

    # Load originals
    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    originals = []
    for p in image_files[:n]:
        img = Image.open(p).convert('RGB')
        originals.append(tfm(img))
    originals = torch.stack(originals, dim=0)
    originals_vis = ((originals + 1) / 2).clamp(0, 1)

    # Load decoder
    if not checkpoint.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    model = ImprovedFeatureToImageDecoder(feature_dim=feats.size(1), img_size=args.img_size).to(device)
    try:
        ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(str(checkpoint), map_location=device)
    state_dict = ckpt.get('generator', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        gen = model(feats)
        gen_vis = denorm(gen)

    # Save grid and per-pair images
    grid_path = save_dir / 'decoder_pairs_grid.png'
    visualize_pairs_grid(originals_vis, gen_vis, grid_path, cols=args.pairs_grid_cols)

    for i in range(n):
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(originals_vis[i].permute(1, 2, 0).cpu().numpy()); axes[0].axis('off'); axes[0].set_title('Original')
        axes[1].imshow(gen_vis[i].permute(1, 2, 0).cpu().numpy()); axes[1].axis('off'); axes[1].set_title('Generated (Normalized)')
        plt.tight_layout()
        plt.savefig(save_dir / f'decoder_pair_{i:04d}.png', dpi=150)
        plt.close()

    print(f'Saved grid: {grid_path}')
    print(f'Saved {n} pair images to: {save_dir}')


if __name__ == '__main__':
    main()