import argparse
from pathlib import Path
import sys
import os
import json
import csv
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Import decoder architecture from local module
try:
    from .decoder_arch import ImprovedFeatureToImageDecoder
except Exception:
    this_dir = Path(__file__).resolve().parent
    if str(this_dir) not in sys.path:
        sys.path.insert(0, str(this_dir))
    from decoder_arch import ImprovedFeatureToImageDecoder


def denorm(t: torch.Tensor) -> torch.Tensor:
    """Map [-1, 1] -> [0, 1] and clamp."""
    return ((t + 1) / 2).clamp(0, 1)


def collect_images(images_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(Path(root) / f)
    return sorted(paths)


def read_image_list(list_path: Path, images_dir: Path) -> List[Path]:
    """Read a text file containing one image path per line. Lines can be absolute
    or relative to images_dir. Empty lines and comments (#) are ignored.
    """
    lines: List[str] = []
    with open(list_path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
    paths: List[Path] = []
    for s in lines:
        p = Path(s)
        if not p.is_absolute():
            p = images_dir / p
        paths.append(p)
    return paths


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(a, b, reduction="mean")


def mae(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(a, b, reduction="mean")


def psnr(a01: torch.Tensor, b01: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """PSNR computed on [0,1] range tensors."""
    m = F.mse_loss(a01, b01, reduction="mean")
    return 10.0 * torch.log10((1.0 ** 2) / (m + eps))


def cosine_similarity_images(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity on flattened image tensors.
    Expects both tensors in the same normalization (we'll use [-1,1]).
    Returns a scalar tensor.
    """
    a_flat = a.view(a.size(0), -1)
    b_flat = b.view(b.size(0), -1)
    # Normalize to unit vectors to avoid scale issues
    a_norm = F.normalize(a_flat, p=2, dim=1)
    b_norm = F.normalize(b_flat, p=2, dim=1)
    cos = (a_norm * b_norm).sum(dim=1)
    return cos


def try_import_skimage():
    try:
        from skimage.metrics import structural_similarity as ssim_fn  # type: ignore
        return ssim_fn
    except Exception:
        return None


def try_init_face_embedder(device: torch.device):
    """Try to initialize a face embedding network (facenet-pytorch).
    Returns a callable model or None if unavailable.
    """
    try:
        from facenet_pytorch import InceptionResnetV1  # type: ignore
        model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        return model
    except Exception:
        return None


def compute_face_embedding_cosine_mean(
    embedder,
    orig01: torch.Tensor,
    gen01: torch.Tensor,
    device: torch.device,
) -> tuple[list[float], float]:
    """Compute cosine similarity between face embeddings of originals and generated images.
    Inputs should be [N,3,H,W] in [0,1]. If embedder is None, returns NaNs.
    """
    n = orig01.size(0)
    if embedder is None or n == 0:
        vals = [float('nan')] * n
        return vals, float('nan')

    # Facenet expects 160x160; resize and normalize to [-1,1]
    resize = transforms.Resize((160, 160))
    with torch.no_grad():
        o = resize(orig01)
        g = resize(gen01)
        o = (o - 0.5) / 0.5
        g = (g - 0.5) / 0.5
        e_o = embedder(o.to(device))
        e_g = embedder(g.to(device))
        e_o = F.normalize(e_o, p=2, dim=1)
        e_g = F.normalize(e_g, p=2, dim=1)
        cos = (e_o * e_g).sum(dim=1).detach().cpu().numpy().astype(float).tolist()
        mean_val = float(np.mean(np.array(cos, dtype=float))) if len(cos) > 0 else float('nan')
    return cos, mean_val


def try_compute_fid(orig01: torch.Tensor, gen01: torch.Tensor, save_dir: Path, device: torch.device) -> float:
    """Compute FID between original and generated images using torch-fidelity if available.
    Saves temporary PNGs under save_dir/for_fid and returns FID score (lower is better).
    Returns NaN if torch-fidelity is unavailable.
    """
    try:
        import torch_fidelity  # type: ignore
    except Exception:
        return float('nan')

    # Prepare temp dirs
    import shutil
    fid_root = save_dir / 'for_fid'
    real_dir = fid_root / 'real'
    fake_dir = fid_root / 'fake'
    if fid_root.exists():
        try:
            shutil.rmtree(fid_root)
        except Exception:
            pass
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    # Save images as PNGs
    n = min(orig01.size(0), gen01.size(0))
    for i in range(n):
        o = (orig01[i].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        g = (gen01[i].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(o).save(real_dir / f"{i:06d}.png")
        Image.fromarray(g).save(fake_dir / f"{i:06d}.png")

    try:
        metrics = torch_fidelity.calculate_metrics(
            input1=str(real_dir),
            input2=str(fake_dir),
            cuda=torch.cuda.is_available() and (device.type == 'cuda'),
            isc=False, kid=False, prc=False,
            fid=True,
            verbose=False,
        )
        fid = float(metrics.get('frechet_inception_distance', float('nan')))
    except Exception:
        fid = float('nan')

    # Best-effort cleanup to save space
    try:
        shutil.rmtree(fid_root)
    except Exception:
        pass
    return fid


def compute_ssim_per_image(orig01: torch.Tensor, gen01: torch.Tensor) -> float:
    """Compute SSIM per-image using skimage if available, else return NaN.
    Inputs must be [C,H,W] in [0,1].
    """
    ssim_fn = try_import_skimage()
    if ssim_fn is None:
        return float("nan")
    # Convert to HWC numpy float32
    o = orig01.permute(1, 2, 0).detach().cpu().numpy()
    g = gen01.permute(1, 2, 0).detach().cpu().numpy()
    # Newer skimage uses channel_axis instead of multichannel
    try:
        val = float(ssim_fn(o, g, data_range=1.0, channel_axis=-1))
    except TypeError:
        val = float(ssim_fn(o, g, data_range=1.0, multichannel=True))
    return val


def try_init_lpips(device: torch.device):
    try:
        import lpips  # type: ignore
        lpips_net = lpips.LPIPS(net='alex').to(device)
        lpips_net.eval()
        return lpips_net
    except Exception:
        return None


def compute_lpips(lpips_net, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute LPIPS on [-1,1] normalized images, shape [N,3,H,W]. Returns [N] tensor.
    If lpips_net is None, returns NaNs.
    """
    if lpips_net is None:
        return torch.full((a.size(0),), float("nan"), device=a.device)
    with torch.no_grad():
        # lpips returns shape [N,1,1,1]; reduce to [N]
        val = lpips_net(a, b)
        return val.view(val.size(0))


def evaluate(
    features_path: Path,
    images_dir: Path,
    checkpoint: Path,
    save_dir: Path,
    num_samples: int,
    img_size: int,
    device: torch.device,
    image_list: Path | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

    save_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")
    features = np.load(str(features_path))
    if features.ndim != 2:
        raise ValueError(f"Expected features [N,D], got {features.shape}")

    # Collect images
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if image_list is not None:
        if not image_list.exists():
            raise FileNotFoundError(f"Image list not found: {image_list}")
        image_files = read_image_list(image_list, images_dir)
    else:
        image_files = collect_images(images_dir)
    if len(image_files) == 0:
        raise RuntimeError(f"No images found in {images_dir}")

    # Log counts and basic sanity
    print(f"Found {len(features)} feature rows and {len(image_files)} images.")
    if len(features) != len(image_files) and num_samples == 0:
        print("[WARN] Number of features and images differ; using min count. Consider providing --image-list to ensure exact alignment.")

    # Align first N samples
    n = min(num_samples if num_samples > 0 else len(features), len(features), len(image_files))
    feats = torch.from_numpy(features[:n]).float().to(device)

    # Load originals with the same preprocessing used during training/visualization
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),  # [-1,1]
    ])
    originals = []
    for p in image_files[:n]:
        img = Image.open(p).convert("RGB")
        originals.append(tfm(img))
    originals = torch.stack(originals, dim=0).to(device)  # [-1,1]

    # Show first few image paths for verification
    print("First few image paths used for evaluation (up to 5):")
    for sp in [str(x) for x in image_files[:5]]:
        print(f"  {sp}")

    # Build and load decoder
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = ImprovedFeatureToImageDecoder(feature_dim=feats.size(1), img_size=img_size).to(device)
    try:
        ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(str(checkpoint), map_location=device)
    state_dict = ckpt.get('generator', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        gen = model(feats)  # [-1,1]

    # Prepare versions for metrics
    # Cosine sim on normalized [-1,1] (scale-invariant on flatten)
    cos_vals = cosine_similarity_images(originals, gen).detach().cpu().numpy().tolist()

    # For pixel-wise metrics, use [0,1]
    orig01 = denorm(originals)
    gen01 = denorm(gen)

    mse_vals: List[float] = []
    mae_vals: List[float] = []
    psnr_vals: List[float] = []
    ssim_vals: List[float] = []

    for i in range(n):
        o = orig01[i:i+1]
        g = gen01[i:i+1]
        mse_vals.append(float(mse(o, g).item()))
        mae_vals.append(float(mae(o, g).item()))
        psnr_vals.append(float(psnr(o, g).item()))
        ssim_vals.append(float(compute_ssim_per_image(orig01[i], gen01[i])))

    # Optional LPIPS on [-1,1]
    lpips_net = try_init_lpips(device)
    lpips_vals: List[float] = []
    if lpips_net is not None:
        lpips_tensor = compute_lpips(lpips_net, originals, gen)
        lpips_vals = lpips_tensor.detach().cpu().numpy().astype(float).tolist()
    else:
        lpips_vals = [float("nan")] * n

    # Optional Identity Preservation via face embeddings
    face_embedder = try_init_face_embedder(device)
    face_embed_cos_vals: List[float] = []
    id_pres_mean: float = float('nan')
    try:
        face_embed_cos_vals, id_pres_mean = compute_face_embedding_cosine_mean(face_embedder, orig01, gen01, device)
    except Exception:
        # keep NaN defaults
        face_embed_cos_vals = [float('nan')] * n
        id_pres_mean = float('nan')

    # Optional Domain Realism via FID (lower is better)
    try:
        fid_score = try_compute_fid(orig01, gen01, save_dir, device)
    except Exception:
        fid_score = float('nan')

    # Aggregate
    def safe_mean(xs: List[float]) -> float:
        arr = np.array(xs, dtype=float)
        if np.isnan(arr).all():
            return float("nan")
        return float(np.nanmean(arr))

    results_per_image: List[Dict[str, Any]] = []
    for i in range(n):
        results_per_image.append({
            "index": i,
            "image_path": str(image_files[i]),
            "cosine": float(cos_vals[i]),
            "mse": float(mse_vals[i]),
            "mae": float(mae_vals[i]),
            "psnr": float(psnr_vals[i]),
            "ssim": float(ssim_vals[i]),
            "lpips": float(lpips_vals[i]),
            "face_embed_cosine": float(face_embed_cos_vals[i]),
        })

    summary: Dict[str, Any] = {
        "num_samples": n,
        "cosine_mean": safe_mean([r["cosine"] for r in results_per_image]),
        "mse_mean": safe_mean([r["mse"] for r in results_per_image]),
        "mae_mean": safe_mean([r["mae"] for r in results_per_image]),
        "psnr_mean": safe_mean([r["psnr"] for r in results_per_image]),
        "ssim_mean": safe_mean([r["ssim"] for r in results_per_image]),
        "lpips_mean": safe_mean([r["lpips"] for r in results_per_image]),
        "identity_preservation_mean": float(id_pres_mean),
        "fid": float(fid_score),
    }

    # Save CSV and JSON
    csv_path = save_dir / "decoder_eval_metrics.csv"
    json_path = save_dir / "decoder_eval_summary.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index", "image_path", "cosine", "mse", "mae", "psnr", "ssim", "lpips", "face_embed_cosine"
            ],
        )
        writer.writeheader()
        for row in results_per_image:
            writer.writerow(row)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return results_per_image, summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate decoder outputs against ground-truth images with multiple metrics")
    parser.add_argument('--features-path', type=str, default=str(Path('model_visuals_ganmodel') / 'normalized_features.npy'))
    parser.add_argument('--images-dir', type=str, default=str(Path('KinFaceW-II') / 'images'))
    parser.add_argument('--checkpoint', type=str, default=str(Path(__file__).resolve().parents[1] / 'decoder_model' / 'decoder_model.pth'))
    parser.add_argument('--save-dir', type=str, default=str(Path('model_visuals_ganmodel') / 'eval'))
    parser.add_argument('--num-samples', type=int, default=0, help='0 means use all aligned samples')
    parser.add_argument('--img-size', type=int, default=64)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--image-list', type=str, default='', help='Optional text file with exact image order used during feature extraction')
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # Resolve project root and paths
    project_root = Path(__file__).resolve().parents[1]
    features_path = Path(args.features_path)
    images_dir = Path(args.images_dir)
    checkpoint = Path(args.checkpoint)
    save_dir = Path(args.save_dir)
    image_list = Path(args.image_list) if args.image_list else None
    if not features_path.is_absolute():
        features_path = project_root / features_path
    if not images_dir.is_absolute():
        images_dir = project_root / images_dir
    if not checkpoint.is_absolute():
        checkpoint = project_root / checkpoint
    if not save_dir.is_absolute():
        save_dir = project_root / save_dir

    results, summary = evaluate(
        features_path=features_path,
        images_dir=images_dir,
        checkpoint=checkpoint,
        save_dir=save_dir,
        num_samples=args.num_samples,
        img_size=args.img_size,
        device=device,
        image_list=image_list,
    )

    print("Evaluation Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Saved per-image metrics CSV to: {save_dir / 'decoder_eval_metrics.csv'}")
    print(f"Saved summary JSON to: {save_dir / 'decoder_eval_summary.json'}")


if __name__ == '__main__':
    main()
