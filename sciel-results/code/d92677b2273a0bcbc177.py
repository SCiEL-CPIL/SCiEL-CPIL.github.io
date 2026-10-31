"""Improved program: Best-Fit-Decreasing style heuristic with adaptive scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We track some state to allow a more sophisticated strategy.
# However, since priority() is called per-item and must be stateless or use only inputs,
# we rely on the bins array which reflects current state.

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for placing the given item.
    bins: array of remaining capacities.
    Returns: array of scores, higher is better.
    Only bins with enough capacity are scored; others get -inf.
    """
    remaining = bins
    # Mask bins that can fit the item
    mask = remaining >= item
    
    # If no bin can fit the item, return -inf for all (shouldn't happen in valid pack, but safe)
    if not np.any(mask):
        return -np.inf * np.ones_like(remaining)
    
    # Estimate the expected next item size from the current item and bin distribution.
    # For Weibull distributions, small items tend to be followed by small items.
    # We use a simple heuristic: expected_next = item * (1 + 0.1 * (1 - item/100))
    # This assumes that if the current item is small, the next is likely also small.
    
    # More robust: use the median of remaining capacities as a proxy for typical item size
    valid_remaining = remaining[mask]
    if len(valid_remaining) > 0:
        # Estimate expected next item as a fraction of the current item, adjusted by bin fill level
        # If bins are mostly full, next item is likely small
        avg_fill = 1.0 - np.mean(valid_remaining) / 100.0  # assuming capacity 100
        expected_next = item * (0.5 + 0.5 * avg_fill)
    else:
        expected_next = item
    
    # Effective gap: gap after placing current item, adjusted for expected next item
    # If we place item in bin with remaining R, new remaining is R - item.
    # We want this to be close to expected_next, so we can fit the next item well.
    # Score should favor bins where (R - item) is close to expected_next.
    
    # Use a smooth score: -((R - item - expected_next)^2 + (R - item))
    # This penalizes both deviation from expected_next and large absolute gaps.
    
    gap = valid_remaining - item
    deviation = gap - expected_next
    
    # Combine: minimize deviation^2 + gap (to avoid large gaps)
    # Score = -(deviation^2 + gap)
    # But we need to normalize to make it comparable across bins.
    
    # Let's use a weighted combination:
    # score = -alpha * deviation^2 - beta * gap
    # With alpha and beta tuned.
    
    alpha = 0.1
    beta = 1.0
    
    scores_valid = -(alpha * deviation**2 + beta * gap)
    
    # Create full score array
    scores = -np.inf * np.ones_like(remaining)
    scores[mask] = scores_valid
    
    return scores

# EVOLVE-BLOCK-END
