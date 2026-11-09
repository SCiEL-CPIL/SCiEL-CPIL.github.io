"""Refined portfolio: Harmonic Mean scoring for resource conservation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin with enough remaining capacity.
    Uses a Harmonic Mean based score to prioritize placements that minimize
    relative waste, acting as a non-linear Best Fit.
    
    Score = - (2 * R * item) / (R + item)
    where R is the remaining capacity after placing the item.
    Higher score (less negative) corresponds to smaller R (tighter fit).
    """
    eps = 1e-9
    
    # Calculate remaining capacity for each bin
    remaining = bins - item
    
    # Mask valid bins (remaining >= 0)
    valid_mask = remaining >= -eps
    
    # Initialize scores with inf (invalid)
    scores = np.full_like(bins, np.inf, dtype=float)
    
    if np.any(valid_mask):
        # Extract valid remaining capacities
        r_valid = remaining[valid_mask]
        
        # Calculate Harmonic Mean of r_valid and item
        # H = (2 * r * item) / (r + item)
        # To avoid division by zero, we use np.maximum on denominator
        denom = r_valid + item
        # If item is 0, H is 0. If r is 0, H is 0.
        h_scores = (2.0 * r_valid * item) / np.maximum(denom, eps)
        
        # We want to MAXIMIZE the score.
        # Best Fit prefers small R. H increases with R.
        # So we negate H to make small R -> less negative -> higher score.
        scores[valid_mask] = -h_scores
        
    return scores
# EVOLVE-BLOCK-END
