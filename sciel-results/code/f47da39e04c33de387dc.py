import numpy as np

def heuristics(prizes, weights):
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle potential shape issues
    if prizes.ndim == 1:
        prizes = prizes.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
        
    # Calculate total prize per item
    total_prize = np.sum(prizes, axis=0)
    
    # Normalize prize to [0, 1] to avoid scale issues
    max_prize = np.max(total_prize)
    if max_prize > 0:
        norm_prize = total_prize / max_prize
    else:
        norm_prize = np.ones_like(total_prize)
        
    # Calculate prize-to-weight ratios for each constraint
    # For each item j and constraint k, compute prize_j / weight_kj
    # Using harmonic mean of these ratios penalizes the bottleneck constraint more heavily
    # This guides ants away from items that risk violating the tightest constraint early
    
    # Add small epsilon to avoid division by zero
    eps = 1e-9
    safe_weights = weights + eps
    
    # Compute prize-to-weight ratio for each constraint
    # Shape: (5, 100)
    ratios = prizes / safe_weights
    
    # Harmonic mean: n / sum(1/ratio)
    # This gives more weight to the smallest ratio (bottleneck constraint)
    inv_ratios = 1.0 / ratios
    sum_inv_ratios = np.sum(inv_ratios, axis=0)
    
    # Avoid division by zero in harmonic mean
    sum_inv_ratios = np.maximum(sum_inv_ratios, eps)
    
    # Number of constraints
    n_constraints = prizes.shape[0]
    
    # Harmonic mean of ratios
    harmonic_mean_ratio = n_constraints / sum_inv_ratios
    
    # Heuristic: normalized prize * harmonic mean of prize-to-weight ratios
    # This emphasizes items with good ratios across ALL constraints, not just the best
    heuristic = norm_prize * harmonic_mean_ratio
    
    # Ensure finite and positive
    heuristic = np.where(np.isfinite(heuristic), heuristic, eps)
    heuristic = np.maximum(heuristic, eps)
    
    return heuristic
