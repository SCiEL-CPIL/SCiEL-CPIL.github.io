"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state to track bin usage statistics across calls
_bin_fill_history = None  # Will be initialized on first call
_capacity = 100.0
_start_time = None


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity. Higher score is better.
    The item goes to the highest-scoring bin.
    
    Uses First-Fit-Diminishing heuristic:
    Score = -remaining capacity (pure best-fit)
    This aggressively consolidates items into bins with least remaining space,
    keeping other bins empty for larger future arrivals.
    """
    global _start_time, _capacity
    
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're getting close to the 150s limit, return quickly
    elapsed = time.time() - _start_time
    if elapsed > 145.0:
        # Emergency: just return best-fit scores quickly
        remaining = bins - item
        valid_mask = remaining >= 0
        scores = np.full_like(remaining, -np.inf, dtype=float)
        if np.any(valid_mask):
            scores[valid_mask] = -remaining[valid_mask]
        return scores
    
    # Track remaining capacity for each bin
    remaining = bins - item
    
    # Filter to only bins that can hold the item
    valid_mask = remaining >= 0
    
    # Create score array with -inf for invalid bins
    scores = np.full_like(remaining, -np.inf, dtype=float)
    
    if not np.any(valid_mask):
        return scores
    
    valid_remaining = remaining[valid_mask]
    n_valid = len(valid_remaining)
    
    if n_valid == 0:
        return scores
    
    # If only one valid bin, return a neutral score
    if n_valid == 1:
        scores[valid_mask] = 0.0
        return scores
    
    # First-Fit-Diminishing: score is negative of remaining capacity
    # This selects the bin with the least remaining space (tightest fit)
    # The rationale: by always picking the tightest fit, we consolidate items
    # into fewer bins, keeping more bins empty for larger future items.
    
    # Primary score: negative remaining capacity (best-fit)
    valid_scores = -valid_remaining
    
    # Adaptive penalty based on item size relative to capacity
    # For small items, we want to be more aggressive with best-fit (consolidate)
    # For large items, we want to be more careful (avoid wasting space)
    item_ratio = item / _CAPACITY
    
    # Use a smooth surrogate: the optimal remaining space after placing the item
    # should be as close to 0 (full bin) or as close to capacity (empty bin) as possible.
    # We model this with a quadratic penalty around the "ideal" remaining space.
    # For best-fit, ideal remaining is 0, but we add a small penalty for being too tight
    # to account for future items that might not fit.
    
    # The penalty encourages remaining space to be either very small (tight fit)
    # or very large (empty bin), with a smooth minimum in between.
    # We use: penalty = -alpha * (valid_remaining - beta * capacity)^2
    # where beta is the "ideal" fraction of capacity to leave empty.
    
    # For small items (item_ratio < 0.1), we want tight fit (beta close to 0)
    # For large items (item_ratio > 0.5), we want to preserve space (beta closer to 0.5)
    beta = 0.05 + 0.15 * item_ratio  # Adaptive ideal remaining fraction
    
    # Weight the penalty more strongly for smaller items to encourage consolidation
    alpha = 0.03 + 0.04 * (1.0 - item_ratio)  # Higher weight for small items
    
    # Smooth quadratic penalty
    ideal_remaining = beta * _CAPACITY
    penalty = -alpha * (valid_remaining - ideal_remaining) ** 2 / _CAPACITY
    
    valid_scores += penalty
    
    # Add a tiny deterministic tie-breaker to ensure reproducibility
    # The tie-breaker is so small it doesn't affect the primary decision
    tie_breaker = np.linspace(0, 1e-9, n_valid)
    valid_scores += tie_breaker
    
    scores[valid_mask] = valid_scores
    
    return scores


# Pre-compute some parameters that might be useful
# The capacity is given as 100 in the task description
_CAPACITY = 100.0

# EVOLVE-BLOCK-END
