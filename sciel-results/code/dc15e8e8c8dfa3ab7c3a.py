"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the item.
    Higher score means more preferred.

    Strategy: A smooth best-fit heuristic that balances:
    1. Minimizing wasted space (best-fit component)
    2. A slight preference for bins that are more filled (to consolidate)
    3. Avoiding bins that are too full (to leave room for future items)

    We use a composite score based on:
    - Remaining capacity after placing the item (lower is better for best-fit)
    - A smooth penalty that discourages leaving very small slivers
    - A mild bonus for bins that are already partially filled
    """
    # Filter bins that have enough remaining capacity
    remaining = bins - item
    valid_mask = remaining >= 0

    # Create a large negative score for invalid bins so they are never chosen
    scores = np.full_like(remaining, -1e9, dtype=np.float64)

    if not np.any(valid_mask):
        return scores

    # Work only on valid bins
    valid_remaining = remaining[valid_mask]

    # Best-fit: prefer bins with least remaining capacity
    # But use a smooth version: -log(remaining + eps) to avoid extreme values
    # This gives diminishing returns as remaining capacity decreases
    eps = 1e-6
    best_fit_score = -np.log(valid_remaining + eps)

    # Add a mild consolidation bonus: bins that are more filled get a small boost
    # This encourages packing items together to leave fresh bins
    # Scale: if bins are around 100, a bin with 50 used gets a small bonus
    fill_ratio = 1.0 - (valid_remaining / (bins[valid_mask] + eps))
    consolidation_bonus = 0.1 * fill_ratio

    # Add a penalty for bins that are very full (remaining < 10% of capacity)
    # This avoids filling bins to the brim which could be suboptimal
    capacity = bins[valid_mask]
    very_full_penalty = np.where(valid_remaining < 0.1 * capacity, -0.5, 0.0)

    # Combine scores
    final_scores = best_fit_score + consolidation_bonus + very_full_penalty

    scores[valid_mask] = final_scores

    return scores
# EVOLVE-BLOCK-END
