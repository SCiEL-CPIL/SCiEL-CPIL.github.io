"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing.
    
    Uses a smooth surrogate approach:
    - Primary: Best-fit decreasing (prefer bin with least remaining space)
    - Secondary: Penalize bins where remaining space is very small relative to item
    - Tertiary: Slight preference for larger bins to reduce fragmentation
    
    The scoring uses a smooth function that balances these objectives.
    """
    rem = bins - item
    scores = np.full_like(rem, -np.inf, dtype=np.float64)
    valid = rem >= 0
    
    if not np.any(valid):
        return scores
    
    valid_rem = rem[valid]
    n_valid = len(valid_rem)
    
    if n_valid == 1:
        scores[valid] = 0.0
        return scores
    
    # Normalize remaining capacity to [0, 1]
    min_rem = np.min(valid_rem)
    max_rem = np.max(valid_rem)
    range_rem = max_rem - min_rem
    
    if range_rem > 1e-9:
        norm_rem = (valid_rem - min_rem) / range_rem
    else:
        norm_rem = np.zeros_like(valid_rem)
    
    # Component 1: Best-fit - strongly prefer bins with least remaining space
    # Use a smooth penalty that increases as remaining space decreases
    # This is like a soft-min: we want to minimize remaining, but smoothly
    best_fit_component = -norm_rem
    
    # Component 2: Fragmentation penalty - penalize bins where remaining is very small
    # compared to item size, to avoid creating unusable fragments
    # Use a smooth function: penalty increases when rem/item is small
    if item > 1e-9:
        ratio = valid_rem / item
        # Smooth penalty: use log(1 + ratio) which is small when ratio is small
        # Actually, we want to AVOID small ratios, so we want to PENALIZE them
        # Let's use: penalty = -log(1 + ratio) which is more negative for small ratio
        # But we want to AVOID small ratios, so we should add this to score (lower is worse)
        # Wait, higher score is better. So we want bins with larger ratio to have higher score.
        # Let's use: bonus = log(1 + ratio) which is larger for larger ratio
        frag_bonus = np.log1p(ratio)
        # Normalize frag_bonus to [0, 1]
        min_frag = np.min(frag_bonus)
        max_frag = np.max(frag_bonus)
        if max_frag > min_frag + 1e-9:
            norm_frag = (frag_bonus - min_frag) / (max_frag - min_frag)
        else:
            norm_frag = np.zeros_like(frag_bonus)
    else:
        norm_frag = np.zeros_like(valid_rem)
    
    # Component 3: Slight worst-fit preference to spread load
    worst_fit_component = norm_rem * 0.1  # Small weight
    
    # Combine: primarily best-fit, with fragmentation bonus and small worst-fit
    # We want: high score = good bin
    # Best-fit: prefer low remaining -> -norm_rem gives higher score for lower remaining
    # Frag bonus: prefer larger ratio -> norm_frag gives higher score for larger ratio
    # Worst-fit: slight preference for larger remaining -> small positive weight
    
    combined = 0.7 * best_fit_component + 0.2 * norm_frag + worst_fit_component
    
    scores[valid] = combined
    return scores
# EVOLVE-BLOCK-END
