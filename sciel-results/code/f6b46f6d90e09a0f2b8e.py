import numpy as np

def heuristics(prizes, weights):
    """
    Returns a heuristic array of shape (100,) for the MCKP.
    Uses a blend of sum-ratio and geo-ratio heuristics.
    """
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Clip values to prevent division by zero and log of zero
    eps = 1e-9
    p = np.clip(prizes, eps, None)
    w = np.clip(weights, eps, None)
    
    # Sum-ratio heuristic: total prize / total weight
    sum_ratio = p.sum(axis=0) / w.sum(axis=0)
    
    # Per-constraint ratios: shape (5, 100)
    ratios = p / w
    
    # Geometric mean of ratios
    # log(geo_mean) = mean(log(ratios))
    log_ratios = np.log(ratios)
    mean_log_ratio = np.mean(log_ratios, axis=0)
    geo_ratio = np.exp(mean_log_ratio)
    
    # Normalize both heuristics to have similar scale
    # Use min-max normalization to [0, 1]
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val > eps:
            return (arr - min_val) / (max_val - min_val)
        return np.ones_like(arr) / arr.size
    
    sum_ratio_norm = normalize(sum_ratio)
    geo_ratio_norm = normalize(geo_ratio)
    
    # Blend with 50-50 weight
    heuristic = 0.5 * sum_ratio_norm + 0.5 * geo_ratio_norm
    
    # Scale back up to avoid being too small (ACO uses pheromone * heuristic)
    heuristic = heuristic + 1e-9
    
    # Ensure all values are positive finite
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e9, neginf=1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
