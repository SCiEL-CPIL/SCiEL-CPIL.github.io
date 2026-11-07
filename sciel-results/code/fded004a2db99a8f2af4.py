import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    demands = np.array(demands, dtype=float)
    
    # Compute residual space for all pairs
    # R[i, j] = capacity - demands[i] - demands[j]
    R = capacity - (demands[:, None] + demands[None, :])
    
    # Initialize heuristic matrix with a small value for invalid pairs
    H = np.full((n, n), 1e-9)
    
    # Find valid pairs where R >= 0
    valid_mask = R >= 0
    
    if np.any(valid_mask):
        # Sort demands to use searchsorted for counting
        sorted_demands = np.sort(demands)
        
        # For each valid pair, count how many items in the full set fit in the residual R
        counts = np.searchsorted(sorted_demands, R, side='right')
        
        # Heuristic components:
        # 1. Pair sum: larger sums are better (more capacity used)
        pair_sum = demands[:, None] + demands[None, :]
        
        # 2. Dead gap bonus: if residual is below min demand, it's a perfect dead gap
        min_demand = np.min(demands)
        # Smooth indicator: 1 when R < min_demand, decays as R grows
        # Use exponential decay: exp(-R / scale) where scale is small
        scale = min_demand * 0.5
        dead_gap_score = np.exp(-np.maximum(R, 0) / scale)
        
        # 3. Inverse count penalty: penalize leaving room for many items
        # Use (count + 1) in denominator to avoid division by zero
        count_penalty = 1.0 / (counts + 1)
        
        # Combine: primary signal is dead gap, secondary is pair sum / count
        # Weight the dead gap heavily since it directly measures waste
        H_valid = pair_sum * (0.5 * dead_gap_score + 0.5 * count_penalty)
        
        # Apply mask
        H[valid_mask] = H_valid[valid_mask]
        
        # Ensure no NaNs or Infs
        H = np.nan_to_num(H, nan=1e-9, posinf=1e-9, neginf=1e-9)
        
    return H
