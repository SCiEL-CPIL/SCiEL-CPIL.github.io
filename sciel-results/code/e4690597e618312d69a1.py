"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for adaptive parameters
_start_time = time.time()
_deadline = 145.0  # seconds (safety margin before 150s kill)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score = better choice.
    
    Combines the current program's "danger zone" penalty (penalizing gaps
    that are too small for typical future items) with a smooth worst-fit
    bonus (from inspiration 1) for bins with large remaining capacity.
    
    This creates a non-linear scoring function that:
    1. Strongly prefers tight fits (best-fit) when remaining is small.
    2. Penalizes "awkward" gaps in the middle range (0.05-0.3 of item)
       that are likely wasted space.
    3. Provides a bonus for bins with large remaining capacity (worst-fit),
       preserving them for potentially larger future items.
    
    The combination uses a smooth blend:
    - Base: -remaining (best-fit)
    - Penalty: Gaussian bump centered at rel_gap=0.15 (danger zone)
    - Bonus: Smooth ramp for remaining > 2*item (worst-fit component)
    """
    remaining = bins - item
    valid = remaining >= 0
    
    if not np.any(valid):
        return np.full(len(bins), -np.inf)
    
    # Relative gap: remaining / item
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_gap = np.where(valid, remaining / max(item, 1e-10), 0.0)
    
    # Component 1: Best-fit score (negative remaining, scaled by item for comparability)
    # Using -rel_gap makes the score scale-invariant with respect to item size
    best_fit_score = -rel_gap
    
    # Component 2: Danger zone penalty
    # Gaps between ~0.05 and ~0.3 of item size are problematic
    # They're too small for most future items but create waste
    # Use a smooth Gaussian bump centered at rel_gap=0.15, width=0.12
    center = 0.15
    width = 0.12
    penalty = np.exp(-((rel_gap - center) ** 2) / (2 * width ** 2))
    penalty = np.where(valid, penalty, 0.0)
    
    # Scale penalty weight inversely with item size (capped)
    # For small items, the danger zone is more critical
    item_scale = min(1.0, 1.0 / max(item, 1.0))
    penalty_weight = 0.8 * item_scale
    
    # Component 3: Worst-fit bonus (from inspiration 1)
    # Bonus for bins with large remaining capacity (remaining > 2*item)
    # This helps preserve empty space for larger future items
    # Use a smooth ramp: max(0, remaining - 2*item) / item = max(0, rel_gap - 2)
    worst_fit_bonus = np.where(valid, np.maximum(0.0, rel_gap - 2.0), 0.0)
    
    # Weight for worst-fit component
    # Lower weight than penalty to keep best-fit dominant
    wf_weight = 0.3
    
    # Combine all components
    score = best_fit_score - penalty_weight * penalty + wf_weight * worst_fit_bonus
    
    # Mask invalid bins
    score = np.where(valid, score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
