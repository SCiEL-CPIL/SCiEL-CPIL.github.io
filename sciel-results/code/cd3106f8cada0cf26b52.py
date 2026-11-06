import numpy as np
import time

# We assume capacity is 100 as per task description
CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining space
    remaining = bins - item
    
    # Initialize scores with -inf for invalid bins
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Find valid indices (bins with enough remaining capacity)
    valid_mask = remaining >= 0
    valid_idx = np.where(valid_mask)[0]
    
    if len(valid_idx) == 0:
        return scores
    
    # Use a smooth blending function to transition between best-fit and worst-fit
    # For small items, prefer best-fit (minimize remaining)
    # For large items, prefer worst-fit (maximize remaining)
    # Use a sigmoid-like transition centered around 0.4 * CAPACITY
    center = 0.4 * CAPACITY
    width = 0.15 * CAPACITY
    
    # Compute the blend weight: 0 for pure best-fit, 1 for pure worst-fit
    t = item / CAPACITY  # normalized item size
    weight = 1.0 / (1.0 + np.exp(-(t - center / CAPACITY) / width * CAPACITY))
    
    # For best-fit: score = -remaining (higher is better = less remaining)
    # For worst-fit: score = remaining (higher is better = more remaining)
    # Blend: score = weight * remaining + (1 - weight) * (-remaining)
    #        = (2*weight - 1) * remaining
    blend_factor = 2.0 * weight - 1.0
    
    scores[valid_idx] = blend_factor * remaining[valid_idx]
    
    return scores
