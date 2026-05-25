import numpy as np
import torch
from pathlib import Path
import csv

# TODO: add marigold-dc to requirements.txt and clone the repo
# git clone https://github.com/prs-eth/Marigold-DC
from marigold import MarigoldDepthCompletionPipeline

from eval.metrics import compute_metrics


def load_arkitscenes_sample(scene_dir: Path):
    """
    Load a single ARKitScenes frame: RGB, sparse depth, and Faro GT.
    
    ARKitScenes structure per scene:
        <scene_id>/
            lowres_depth/     <- iPhone LiDAR depth maps (low res, ~256x192)
            highres_color/    <- RGB frames (high res, ~1920x1440)
            laser_scan_5mm/   <- Faro ground truth point cloud
    
    Returns:
        rgb: (H, W, 3) uint8
        sparse_depth: (H, W) float32, subsampled from lowres_depth
        gt_depth: (H, W) float32, from Faro laser scan
    """
    # TODO: implement actual file loading once data is on VM
    # placeholder shapes match ARKitScenes highres dimensions
    raise NotImplementedError("implement after data is mounted on GCP")


def subsample_depth(depth_map, keep_fraction=0.01):
    """
    Simulate sparse LiDAR input by randomly subsampling the low-res depth map.
    
    In practice iPhone LiDAR gives ~hundreds of valid points per frame.
    keep_fraction=0.01 on a 256x192 map → ~500 points, realistic sparsity.
    
    Args:
        depth_map: (H, W) float32 dense depth
        keep_fraction: fraction of valid pixels to keep
    
    Returns:
        sparse: (H, W) float32, zeros everywhere except kept points
    """
    sparse = np.zeros_like(depth_map)
    valid_mask = depth_map > 0
    valid_idx = np.argwhere(valid_mask)

    n_keep = max(1, int(len(valid_idx) * keep_fraction))
    chosen = valid_idx[np.random.choice(len(valid_idx), n_keep, replace=False)]

    sparse[chosen[:, 0], chosen[:, 1]] = depth_map[chosen[:, 0], chosen[:, 1]]
    return sparse


def run_baseline(val_split_path: Path, data_root: Path, output_path: Path):
    """
    Run Marigold-DC on the validation split and save per-frame metrics to CSV.
    
    Args:
        val_split_path: path to splits/val.txt (one scene ID per line)
        data_root: root dir of ARKitScenes data on VM
        output_path: where to save results CSV
    """
    # load marigold-dc pipeline onto GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {device}")
    pipe = MarigoldDepthCompletionPipeline.from_pretrained(
        "prs-eth/marigold-depth-completion-v1-0"
    ).to(device)

    scene_ids = val_split_path.read_text().strip().splitlines()
    print(f"running baseline on {len(scene_ids)} scenes")

    results = []

    for scene_id in scene_ids:
        scene_dir = data_root / scene_id
        try:
            rgb, sparse_depth, gt_depth = load_arkitscenes_sample(scene_dir)
        except NotImplementedError:
            print(f"skipping {scene_id} — loader not implemented yet")
            continue

        # run marigold-dc: takes RGB + sparse depth, returns dense prediction
        pred = pipe(
            image=rgb,
            sparse_depth=sparse_depth,
        ).depth_np  # (H, W) float32

        metrics = compute_metrics(pred, gt_depth)
        results.append({"scene_id": scene_id, **metrics})
        print(f"{scene_id}: absrel={metrics['absrel']:.4f} rmse={metrics['rmse']:.4f}")

    # dump to CSV so you can load into pandas for the slides table
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_id", "rmse", "mae", "absrel", "d1", "d2", "d3"])
        writer.writeheader()
        writer.writerows(results)

    print(f"saved results to {output_path}")

    # print mean across all scenes
    mean_metrics = {k: np.mean([r[k] for r in results]) for k in ["rmse", "mae", "absrel", "d1", "d2", "d3"]}
    print("\n--- mean metrics ---")
    for k, v in mean_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    run_baseline(
        val_split_path=Path("splits/val.txt"),
        data_root=Path("data/arkitscenes"),
        output_path=Path("eval/marigold_dc_results.csv"),
    )
