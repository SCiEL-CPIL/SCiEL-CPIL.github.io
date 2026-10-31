"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score = better choice.
    
    Strategy: Polarization scoring.
    - For small items (< 25): prefer bins that are either very full or very empty
      (polarized), to avoid leaving medium-sized gaps.
    - For large items (>= 25): use best-fit (prefer bins with least remaining space
      after placing the item) to consolidate large items.
    
    The polarization score maximizes distance from half-capacity (50), which:
    - For small items in nearly-full bins: keeps them consolidated
    - For small items in nearly-empty bins: preserves half-full bins for future medium items
    - Avoids creating awkward ~25-75 remaining spaces that are hard to fill
    """
    remaining = bins - item
    # Only bins with enough capacity are valid
    valid = remaining >= 0
    
    # For invalid bins, assign a very low score so they're never chosen
    scores = np.full_like(remaining, -1e9, dtype=np.float64)
    
    if not np.any(valid):
        return scores
    
    valid_remaining = remaining[valid]
    
    if item < 25.0:
        # Polarization: maximize |remaining - 50| to avoid half-full bins
        # This pushes small items either into nearly-full bins (consolidation)
        # or into nearly-empty bins (preserving flexibility)
        scores[valid] = np.abs(valid_remaining - 50.0)
    else:
        # Best-fit for larger items: minimize remaining space after placement
        scores[valid] = -valid_remaining
    
    return scores
# EVOLVE-BLOCK-END
