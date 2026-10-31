"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # 5 seconds margin before 150s

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing.
    Combines polarization (distance from half-capacity) with a smooth best-fit term.
    The polarization strategy prefers remainders near 0 or near capacity,
    while the best-fit term provides a baseline preference for smaller remainders.
    """
    # Deadline check: if we are running out of time, fall back to simple Best Fit
    if time.time() > _DEADLINE:
        remaining = bins - item
        scores = np.full_like(bins, -1e9, dtype=float)
        valid = remaining >= 0
        scores[valid] = -remaining[valid]
        return scores

    # Calculate remaining capacity after placing the item
    remaining = bins - item
    
    # Mask for bins that can fit the item
    valid_mask = remaining >= 0
    
    # Initialize scores to a very low value for invalid bins
    scores = np.full_like(bins, -1e9, dtype=float)
    
    if not np.any(valid_mask):
        return scores

    # Extract valid bins' remaining capacities
    valid_remaining = remaining[valid_mask]
    
    # Capacity is specified as 100 in the task description
    C = 100.0
    half_cap = C / 2.0
    
    # Polarization score: prefer remainders far from half-capacity
    # This encourages creating full bins or keeping bins empty
    polarization_score = np.abs(valid_remaining - half_cap)
    
    # Best-fit score: prefer smaller remainders (negative remaining)
    # This provides a baseline preference for tight fits
    best_fit_score = -valid_remaining
    
    # Combine the two scores with weights
    # We weight polarization more heavily to encourage polarization strategy
    # but keep some best-fit to avoid obviously bad placements
    w_polar = 1.0
    w_best = 0.1
    
    final_scores = w_polar * polarization_score + w_best * best_fit_score
    
    # Assign scores back to the original array
    scores[valid_mask] = final_scores
    
    return scores
# EVOLVE-BLOCK-END
