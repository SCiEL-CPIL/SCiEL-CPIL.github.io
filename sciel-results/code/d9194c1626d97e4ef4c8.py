import numpy as np

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ACO bin packing algorithm.
    
    The heuristic eta[i, j] represents the attractiveness of placing item j
    immediately after item i in the same bin.
    
    Strategy:
    We use a composite heuristic that combines:
    1. Best-fit preference: smaller gap is more attractive (inverse gap)
    2. Fill-ratio signal: higher utilization of the bin is preferred
    3. Item size: larger items are preferred to avoid stranding
    
    Formula: eta[i, j] = (w_j / (gap + eps)) * (1 + alpha * (fill_ratio)^2)
    where gap = capacity - w_i - w_j, fill_ratio = (w_i + w_j) / capacity
    The fill-ratio term provides a smooth secondary signal that rewards
    pairs that fill the bin well, helping ACO escape tight-fit local minima
    while preserving the strong best-fit preference.
    """
    W = np.array(demands, dtype=float)
    n = len(W)
    
    # Calculate sums of all pairs
    pair_sums = W[:, None] + W[None, :]
    
    # Calculate gaps
    gaps = capacity - pair_sums
    
    # Identify valid pairs
    fits = gaps >= 0
    
    # Initialize heuristic matrix with a very small value for invalid pairs
    H = np.full((n, n), 1e-9)
    
    if np.any(fits):
        # Add epsilon to avoid division by zero for exact fits
        eps = 1e-9
        safe_gaps = np.where(fits, gaps + eps, 1.0)
        
        # Numerator: size of the item being placed (j)
        numerator = W[None, :]
        
        # Base best-fit term: w_j / (gap + eps)
        base_term = numerator / safe_gaps
        
        # Fill ratio: how much of the capacity is used by this pair
        # Higher fill ratio means better bin utilization
        fill_ratio = np.where(fits, pair_sums / capacity, 0.0)
        
        # Secondary signal: reward high fill ratios with a smooth quadratic
        # This helps guide ACO toward combinations that fill bins well
        # while the primary inverse-gap term maintains best-fit preference
        alpha = 0.3  # Weight for the fill-ratio bonus
        fill_bonus = 1.0 + alpha * (fill_ratio ** 2)
        
        # Combine: base best-fit multiplied by fill-ratio bonus
        H_valid = base_term * fill_bonus
        
        # Set invalid pairs to 1e-9
        H = np.where(fits, H_valid, 1e-9)
        
    # Final safety checks: ensure finite and positive
    H = np.where(np.isfinite(H), H, 1e-9)
    H = np.where(H > 0, H, 1e-9)
    
    return H
