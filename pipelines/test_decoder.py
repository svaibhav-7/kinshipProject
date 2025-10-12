import os
import argparse
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import sys
from torchvision import transforms

# Reuse the trained decoder architecture (side-effect free)
try:
    # When executed as a module: python -m pipelines.test_decoder
    from .decoder_arch import ImprovedFeatureToImageDecoder
except Exception:
    # When executed as a script: python pipelines/test_decoder.py
    this_dir = Path(__file__).resolve().parent
    if str(this_dir) not in sys.path:
        sys.path.insert(0, str(this_dir))
    from decoder_arch import ImprovedFeatureToImageDecoder


def denormalize_img_tensor(t: torch.Tensor) -> torch.Tensor:
    """Convert tensor in [-1, 1] to [0, 1] and clamp."""
    return ((t + 1) / 2).clamp(0, 1)


def save_tensor_as_image(t: torch.Tensor, path: Path):
    """Save a single image tensor (C,H,W in [0,1]) to disk as PNG."""
    t = t.detach().cpu()
    if t.dim() == 4:
        t = t[0]
    c, h, w = t.shape
    img = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def visualize_grid(tensors: torch.Tensor, save_path: Path, cols: int = 5, titles: list | None = None):
    """Create and save a grid visualization for quick inspection.

    Args:
        tensors: Tensor of shape [N, 3, H, W] in [0,1]
        save_path: Where to save the grid PNG
        cols: number of columns
        titles: optional list of strings (length N) to display above each image
    """
    tensors = tensors.detach().cpu()
    n = tensors.size(0)
    cols = min(cols, n)
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 3, rows * 3))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        img = tensors[i]
        img = img.permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.axis('off')
        if titles and i < len(titles):
            plt.title(titles[i])
        else:
            plt.title(f"#{i}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    try:
        plt.show()
    except Exception:
        pass
    plt.close()


def visualize_pairs_grid(
    originals: torch.Tensor,
    generated: torch.Tensor,
    save_path: Path,
    labels: list | None = None,
    cols: int = 5,
):
    """Create a side-by-side grid with Original (top row) and Generated (bottom row).

    Args:
        originals: [N,3,H,W] in [0,1]
        generated: [N,3,H,W] in [0,1]
        labels: optional list of strings per sample like 'Child'/'Adult'
    """
    n = min(originals.size(0), generated.size(0))
    cols = min(cols, n)
    rows = 2
    plt.figure(figsize=(cols * 3, rows * 3))

    for j in range(cols):
        # Original (top)
        plt.subplot(rows, cols, j + 1)  # positions 1..cols
        o = originals[j].detach().cpu().permute(1, 2, 0).numpy()
        plt.imshow(o)
        plt.axis('off')
        title = "Original"
        if labels and j < len(labels) and labels[j] is not None:
            title += f" ({labels[j]})"
        plt.title(title)

        # Generated (bottom)
        plt.subplot(rows, cols, cols + j + 1)
        g = generated[j].detach().cpu().permute(1, 2, 0).numpy()
        plt.imshow(g)
        plt.axis('off')
        gen_title = "Generated (Normalized)"
        if labels and j < len(labels) and labels[j] is not None:
            gen_title = f"Generated (from {labels[j]})"
        plt.title(gen_title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    try:
        plt.show()
    except Exception:
        pass
    plt.close()


def load_original_images(
    image_paths: list[Path],
    img_size: int,
    limit: int,
) -> torch.Tensor:
    """Load up to 'limit' images as a tensor [N,3,H,W] in [0,1]."""
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    imgs = []
    for p in image_paths[:limit]:
        try:
            img = Image.open(p).convert('RGB')
            t = tfm(img)
            imgs.append(t)
        except Exception:
            continue

    if len(imgs) == 0:
        return torch.empty(0)
    return torch.stack(imgs, dim=0)


def run_inference(
    features_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    num_samples: int,
    img_size: int,
    device: str,
    age_labels_path: Path | None = None,
    images_dir: Path | None = None,
    images_list: Path | None = None,
):
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Resolve paths relative to project root (parent of pipelines)
    project_root = Path(__file__).resolve().parents[1]
    features_path = Path(features_path)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    if not features_path.is_absolute():
        features_path = project_root / features_path
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    # Resolve optional age labels path
    if age_labels_path:
        age_labels_path = Path(age_labels_path)
        if not age_labels_path.is_absolute():
            age_labels_path = project_root / age_labels_path

    # Resolve optional images_dir and images_list
    if images_dir:
        images_dir = Path(images_dir)
        if not images_dir.is_absolute():
            images_dir = project_root / images_dir
    if images_list:
        images_list = Path(images_list)
        if not images_list.is_absolute():
            images_list = project_root / images_list

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load normalized features (expected shape: [N, feature_dim])
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found at: {features_path}")
    features = np.load(str(features_path))
    if features.ndim != 2:
        raise ValueError(f"Expected features to be 2D [N, D], got shape {features.shape}")
    n_total, feature_dim = features.shape
    if n_total == 0:
        raise ValueError("No features found in the provided .npy file")

    # Try to load age labels if provided or if a sidecar file exists
    labels = None
    sidecar_default = features_path.with_name(features_path.stem.replace("normalized_features", "normalized_features_labels") + ".npy")
    try_paths = []
    if age_labels_path is not None:
        try_paths.append(age_labels_path)
    try_paths.append(sidecar_default)
    for p in try_paths:
        if p is not None and p.exists():
            try:
                labels = np.load(str(p))
                if labels.shape[0] != n_total:
                    print(f"[WARN] Age labels count ({labels.shape[0]}) doesn't match features ({n_total}). Ignoring labels.")
                    labels = None
                else:
                    print(f"[INFO] Loaded age labels from: {p}")
                break
            except Exception as e:
                print(f"[WARN] Failed to load age labels from {p}: {e}")

    num = min(num_samples, n_total)
    features = features[:num]
    if labels is not None:
        labels = labels[:num]

    # Build model and load weights
    model = ImprovedFeatureToImageDecoder(feature_dim=feature_dim, img_size=img_size).to(device)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    # PyTorch 2.6 defaults to weights_only=True which can break older checkpoints.
    # Force weights_only=False and fall back for older torch versions that don't accept the arg.
    try:
        ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch: no weights_only arg
        ckpt = torch.load(str(checkpoint_path), map_location=device)

    # Support both plain state_dict and checkpoint dict formats
    state_dict = ckpt.get('generator', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # Run inference
    with torch.no_grad():
        feats_tensor = torch.from_numpy(features).float().to(device)
        fake_imgs = model(feats_tensor)  # in [-1, 1]
        fake_imgs = denormalize_img_tensor(fake_imgs)  # to [0, 1]

    # Save individual images
    saved_paths = []
    titles = []
    for i in range(fake_imgs.size(0)):
        # Build title/filename label
        label_str = None
        if labels is not None:
            if int(labels[i]) == 0:
                label_str = "Child"
            elif int(labels[i]) == 1:
                label_str = "Adult"
        title = f"Gen (from {label_str})" if label_str else "Gen (normalized)"
        titles.append(title)

        fname_prefix = f"recon_{i:04d}"
        if label_str:
            fname_prefix = f"recon_{label_str.lower()}_{i:04d}"
        out_path = output_dir / f"{fname_prefix}.png"
        save_tensor_as_image(fake_imgs[i], out_path)
        saved_paths.append(out_path)

    # Save grid
    grid_path = output_dir / "grid.png"
    visualize_grid(fake_imgs, grid_path, cols=5, titles=titles)

    # If originals available, build side-by-side pairs
    original_paths: list[Path] = []
    if images_list and images_list.exists():
        try:
            with open(images_list, 'r') as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    p = Path(s)
                    if not p.is_absolute():
                        p = (images_dir or project_root) / s if images_dir else project_root / s
                    original_paths.append(p)
        except Exception as e:
            print(f"[WARN] Failed to read images list {images_list}: {e}")
    elif images_dir and images_dir.exists():
        # Walk directory and collect images deterministically
        for root, _, files in os.walk(images_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    original_paths.append(Path(root) / f)
        original_paths = sorted(original_paths)

    originals_tensor = torch.empty(0)
    if len(original_paths) > 0:
        originals_tensor = load_original_images(original_paths, img_size, fake_imgs.size(0))

    if originals_tensor.numel() > 0:
        # Prepare labels for titles
        orig_labels = []
        for i in range(originals_tensor.size(0)):
            if i < len(titles) and "Child" in titles[i]:
                orig_labels.append("Child")
            elif i < len(titles) and "Adult" in titles[i]:
                orig_labels.append("Adult")
            else:
                orig_labels.append(None)

        # Save paired grid
        pairs_grid_path = output_dir / "grid_pairs.png"
        visualize_pairs_grid(originals_tensor, fake_imgs[: originals_tensor.size(0)], pairs_grid_path, labels=orig_labels, cols=pairs_grid_cols)

        # Save per-sample side-by-side images
        for i in range(originals_tensor.size(0)):
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            axes[0].imshow(originals_tensor[i].permute(1, 2, 0).numpy())
            axes[0].axis('off')
            left_title = "Original"
            if orig_labels[i] is not None:
                left_title += f" ({orig_labels[i]})"
            axes[0].set_title(left_title)

            axes[1].imshow(fake_imgs[i].permute(1, 2, 0).numpy())
            axes[1].axis('off')
            right_title = "Generated (Normalized)"
            if orig_labels[i] is not None:
                right_title = f"Generated (from {orig_labels[i]})"
            axes[1].set_title(right_title)

            plt.tight_layout()
            pair_name = f"pair_{orig_labels[i].lower() if orig_labels[i] else 'sample'}_{i:04d}.png"
            plt.savefig(output_dir / pair_name, dpi=150)
            plt.close()

    print(f"Saved {len(saved_paths)} reconstructed images to: {output_dir}")
    print(f"Grid preview saved to: {grid_path}")
    if originals_tensor.numel() > 0:
        print(f"Side-by-side grid saved to: {pairs_grid_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct age-normalized fake images using the trained decoder")
    parser.add_argument(
        "--features-path",
        type=str,
        default=str(Path("model_visuals_ganmodel") / "normalized_features.npy"),
        help="Path to normalized features .npy (from Phase 2)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "decoder_model" / "decoder_model.pth"),
        help="Path to trained decoder checkpoint (.pth)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("recon_results")),
        help="Directory to save reconstructed images",
    )
    parser.add_argument("--num-samples", type=int, default=20, help="Number of first samples to reconstruct")
    parser.add_argument("--img-size", type=int, default=64, help="Output image size expected by decoder")
    parser.add_argument("--device", type=str, default="", help="cuda or cpu (auto if empty)")
    parser.add_argument(
        "--age-labels-path",
        type=str,
        default="",
        help="Optional .npy with age labels per sample (0=child,1=adult). If omitted, will try sidecar file normalized_features_labels.npy",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="",
        help="Optional directory containing original images. Will take first N images (sorted) to pair with features",
    )
    parser.add_argument(
        "--images-list",
        type=str,
        default="",
        help="Optional text file with one image path per line (absolute or relative to project root) to pair with features",
    )
    parser.add_argument(
        "--pairs-grid-cols",
        type=int,
        default=5,
        help="Number of columns to show in the side-by-side pairs grid",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        features_path=Path(args.features_path),
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples,
        img_size=args.img_size,
        device=args.device,
        age_labels_path=Path(args.age_labels_path) if args.age_labels_path else None,
        images_dir=Path(args.images_dir) if args.images_dir else None,
        images_list=Path(args.images_list) if args.images_list else None,
    )
