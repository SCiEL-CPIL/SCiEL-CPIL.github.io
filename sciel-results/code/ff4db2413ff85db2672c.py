"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing.
    Uses a smoothed best-fit heuristic that penalizes very tight fits
    to avoid creating unusable gaps, while still preferring bins that
    leave less remaining capacity overall.
    """
    remaining = bins - item
    valid = remaining >= 0
    
    # For valid bins, compute a score that:
    # 1. Prefers smaller remaining capacity (best-fit tendency)
    # 2. But penalizes extremely small remaining (avoiding unusable gaps)
    # Use a quadratic bump: score = -remaining + penalty * remaining^2
    # The minimum of -r + p*r^2 is at r = 1/(2p), so we prefer remaining
    # near 1/(2p). With p=0.005, preferred remaining ~ 100.
    
    # Actually, let's use a simpler approach:
    # score = -remaining, but add a small bonus for bins with moderate remaining
    # to avoid the "cliff" at zero.
    
    # Let's try: score = -remaining + 0.001 * remaining^2
    # This creates a slight preference for non-zero remaining.
    
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    if np.any(valid):
        r = remaining[valid]
        # Smoothed best-fit: prefer small remaining, but with a quadratic
        # term that softens the preference for very small remaining
        scores[valid] = -r + 0.002 * r * r
    
    return scores
# EVOLVE-BLOCK-END
