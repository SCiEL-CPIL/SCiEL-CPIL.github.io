import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining capacity for each bin
    rem = bins - item
    
    # Valid bins have non-negative remaining capacity
    valid_mask = rem >= 0
    
    # Initialize scores to -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid_mask):
        return scores
    
    valid_rem = rem[valid_mask]
    
    # Adaptive best-fit with smooth surrogate:
    # For Weibull distributions, we want to balance tight packing with
    # avoiding excessive fragmentation. Use a smooth function that
    # penalizes both large remaining space (waste) and very tight fits
    # (which can lead to future fragmentation).
    
    # The ratio item/rem indicates how "tight" the fit is.
    # A ratio close to 1.0 means the item nearly fills the remaining space.
    # We want to favor ratios in a sweet spot, not too loose, not too tight.
    
    # Use a smooth penalty based on deviation from an optimal fit ratio.
    # For Weibull with shape parameter ~1-2, an optimal ratio is around 0.7-0.9.
    # We use a quadratic penalty around 0.8 as a heuristic.
    
    # Compute the fit ratio: item / (item + rem) = item / bins_used
    # This is the fraction of the bin's remaining capacity used by this item.
    bin_used = item + valid_rem  # This is the total capacity of the bin
    fit_ratio = item / np.maximum(bin_used, 1e-10)
    
    # Smooth score: prefer fit_ratio near 0.8, penalize deviation
    # Using a negative quadratic to create a smooth peak at 0.8
    optimal_ratio = 0.8
    smooth_score = -(fit_ratio - optimal_ratio) ** 2
    
    # Also incorporate the absolute remaining space as a secondary criterion
    # to break ties and ensure we still prefer tighter fits when ratios are similar
    # Scale rem to be comparable to the smooth_score range
    rem_penalty = -valid_rem * 0.1
    
    valid_scores = smooth_score + rem_penalty
    
    scores[valid_mask] = valid_scores
    return scores
