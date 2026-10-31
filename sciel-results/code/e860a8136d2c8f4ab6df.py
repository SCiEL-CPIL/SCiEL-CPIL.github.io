"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = None
_START_TIME = None

def _ensure_deadline():
    global _DEADLINE, _START_TIME
    if _DEADLINE is None:
        _START_TIME = time.time()
        _DEADLINE = _START_TIME + 145.0  # 5s safety margin before 150s kill

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    _ensure_deadline()
    
    # Best-fit with smoothing: prefer bins that leave the least remaining capacity,
    # but use a smooth scoring that also considers the ratio of item to remaining space.
    # 
    # For Weibull items, a good heuristic is to prefer bins where the item fills
    # a substantial fraction of the remaining space, while still minimizing waste.
    # 
    # Score = -(remaining) + bonus for good fill ratio
    # where fill ratio = item / (bins + item) for bins that can fit the item
    
    remaining = bins - item
    valid = remaining >= 0
    
    # Start with negative remaining (best-fit)
    scores = np.where(valid, -remaining, -1e18)
    
    # Add a smooth bonus: reward bins where item/bins ratio is in a sweet spot
    # For bins that can fit the item, compute the fill fraction
    # fill = item / bins (how much of the bin this item would occupy relative to current usage)
    # Actually, let's think of it as: after placing, the bin has (bins - remaining) = bins
    # The "tightness" is item / (bins + item) -- no, that's not right.
    
    # Let's use: score = -remaining + alpha * item / (bins + item) for valid bins
    # This rewards both tight packing AND bins where the item is a significant portion
    
    # Simpler: use a weighted combination
    # Primary: minimize remaining (best-fit)
    # Secondary: prefer bins with moderate current fill (not too empty, not too full)
    
    # For valid bins, compute a fill-based bonus
    # fill_fraction = item / bins (how much of the bin's capacity this item uses)
    # We want this to be in a good range, say 0.3-0.8
    
    valid_bins = bins[valid]
    if len(valid_bins) > 0:
        # fill fraction: how much of the bin this item represents
        fill_frac = item / valid_bins
        # Bonus peaks around fill_frac ~ 0.5, use a smooth bump
        # Use: bonus = -abs(fill_frac - 0.5) to prefer moderate fills
        # Or use: bonus = fill_frac * (1 - fill_frac) which peaks at 0.5
        bonus = fill_frac * (1.0 - fill_frac)
        
        # Combine: primary score is -remaining, secondary is bonus
        # Scale bonus relative to remaining
        # Use a small weight for the bonus
        alpha = 0.1
        scores[valid] = -remaining[valid] + alpha * bonus
    
    return scores
# EVOLVE-BLOCK-END
