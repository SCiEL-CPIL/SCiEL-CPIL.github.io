"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing.
    Returns a score array where higher is better.
    Only bins with enough remaining capacity get a finite score; others get -inf.
    
    Strategy: Smooth best-fit with adaptive tau based on item size,
    combined with consolidation bonus for already-used bins.
    Uses a smooth exponential decay with item-size-aware scaling.
    """
    # Deadline guard: track elapsed time for safety
    start_time = time.time()
    DEADLINE = start_time + 115.0  # 5s safety margin before 120s limit
    
    # Ensure bins is a numpy array
    bins = np.asarray(bins, dtype=np.float64)
    
    # Compute remaining capacity after placing item
    remaining = bins - item
    
    # Only consider bins where remaining >= 0 (enough capacity)
    valid_mask = remaining >= 0
    
    # Initialize scores to -inf for invalid bins
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if not np.any(valid_mask):
        return scores
    
    valid_remaining = remaining[valid_mask]
    valid_bins = bins[valid_mask]
    capacity = 100.0
    
    # Adaptive tau: scale with item size for better discrimination
    # For small items, use smaller tau to prefer tight fits
    # For large items, use larger tau to be more lenient
    tau = 5.0 + 15.0 * (item / capacity)
    
    # Primary score: exponential decay of remaining space
    # Smaller tau for small items -> sharper preference for tight fits
    base_score = np.exp(-valid_remaining / tau)
    
    # Consolidation bonus: prefer bins that are already partially used
    # This helps reduce the number of open bins
    # Use a smooth sigmoid-like bonus that increases with fill level
    fill_fraction = 1.0 - (valid_remaining / capacity)
    # Smooth consolidation: tanh-based to avoid discontinuities
    consolidation_bonus = 0.1 * np.tanh(2.0 * (fill_fraction - 0.5))
    
    # Combine scores: primary dominates, consolidation is a smooth modifier
    combined = base_score + consolidation_bonus
    
    # Apply a smooth penalty for very wasteful placements
    # This gently discourages leaving too much space unused
    normalized_remaining = valid_remaining / capacity
    waste_penalty = -0.05 * normalized_remaining ** 2
    
    scores[valid_mask] = combined + waste_penalty
    
    # Check deadline
    if time.time() > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
