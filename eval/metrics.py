import numpy as np

def compute_metrics(pred, gt, min_depth=0.1, max_depth=10.0):
    """
    Compute depth completion metrics between predicted and ground truth depth maps.
    
    Args:
        pred: predicted depth map (H, W), in meters
        gt: ground truth depth map (H, W), in meters
        min_depth: ignore pixels below this depth (default 0.1m)
        max_depth: ignore pixels above this depth (default 10.0m, indoor scenes)
    
    Returns:
        dict of metrics: rmse, mae, absrel, d1, d2, d3
    """
    # mask out invalid GT pixels: zeros, NaNs, and out-of-range values
    # pred is dense so we only mask based on GT validity
    mask = (gt > min_depth) & (gt < max_depth) & (gt > 0) & np.isfinite(gt)
    pred = pred[mask]
    gt = gt[mask]

    # root mean squared error — penalizes large errors heavily
    rmse = np.sqrt(((pred - gt) ** 2).mean())
    
    # mean absolute error — more interpretable, in meters
    mae = np.abs(pred - gt).mean()
    
    # absolute relative error — scale-invariant, main metric for depth
    absrel = (np.abs(pred - gt) / gt).mean()

    # threshold accuracy: % of pixels where max(pred/gt, gt/pred) < threshold
    # d1/d2/d3 at 1.25, 1.25^2, 1.25^3 — higher is better
    thresh = np.maximum(pred / gt, gt / pred)
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    return {
        "rmse": rmse,
        "mae": mae,
        "absrel": absrel,
        "d1": d1,
        "d2": d2,
        "d3": d3,
    }
