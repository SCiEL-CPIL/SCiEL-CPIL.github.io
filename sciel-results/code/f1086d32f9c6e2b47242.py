"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Adaptive Best-Fit with smooth sigmoid transition.
    - For large items (relative to bin capacity), strongly prefer Best-Fit (minimize remaining).
    - For small items, be more neutral to avoid fragmentation (Worst-Fit tendency for small items).
    - This is achieved via a smooth sigmoid that blends between Best-Fit and a neutral score.
    """
    remaining = bins - item
    feasible = remaining >= 0
    
    # Initialize scores with a very low value for infeasible bins
    scores = np.full(len(bins), -1e18)
    
    if not np.any(feasible):
        return scores
    
    # For feasible bins, compute adaptive score based on relative waste minimization
    # Strategy: prioritize bins where the item constitutes a large fraction of remaining capacity
    # This minimizes the "relative waste" of opening/using a bin
    
    # Compute fill ratio: how much of the bin will be filled by this item relative to current load
    # A bin with low current load + item filling most of remaining = good "completion"
    # A bin with high current load + item filling most of remaining = also good (tight fit)
    
    # Relative waste metric: item / (current_load + item) 
    # Higher means item is a larger fraction of the total bin content after placement
    # This naturally favors Best-Fit for large items and is more neutral for small items
    
    # Current load = bins (the remaining capacity array represents unused space)
    # Wait, bins array contains REMAINING capacity, not current load
    # Let me reconsider: bins[i] = remaining capacity in bin i
    # Current load in bin i = capacity - bins[i], but we don't have capacity directly
    # However, we can use the relative position within feasible bins
    
    # Alternative: use the ratio of item to remaining capacity
    # item/remaining is high when item fills most of the remaining space (Best-Fit)
    # item/remaining is low when item is small relative to remaining space (Worst-Fit tendency)
    
    # Smooth approach: score = -log(1 + remaining/item) for item > 0
    # This is a smooth surrogate for Best-Fit that handles the singularity at item->0
    # As item -> 0, log(1 + remaining/item) -> log(remaining/item) -> large, so -log(...) -> very negative
    # This would favor bins with small remaining/item ratio, i.e., Best-Fit even for small items
    # But we want neutral for small items...
    
    # Better: blend between Best-Fit (-remaining) and a neutral score based on item size
    # Use the relative fill: fill_ratio = item / (item + remaining)
    # For large item relative to remaining: fill_ratio -> 1 (strong Best-Fit signal)
    # For small item relative to remaining: fill_ratio -> 0 (weak signal, more neutral)
    
    # Compute fill ratio for feasible bins
    feasible_remaining = remaining[feasible]
    
    # Avoid division by zero
    safe_item = max(item, 1e-10)
    
    # Fill ratio: how much of the bin will be "used up" by this item relative to what's left
    # Actually, let's use: relative_waste = remaining / (item + remaining)
    # Lower relative_waste = item fills more of the remaining space = better
    # This is equivalent to 1 - item/(item + remaining)
    
    # Score: prefer bins with lower relative_waste (item fills more of remaining)
    # But scale by how "confident" we are based on item size
    # Confidence = item / max_bin (normalized item size)
    
    max_bin = np.max(bins)
    if max_bin > 0:
        item_ratio = item / max_bin
    else:
        item_ratio = 0.0
    
    # Confidence weight: sigmoid that is ~1 for large items, ~0.5 for medium, ~0 for small
    # Use a smoother transition centered at 0.3 with k=8
    confidence = 1.0 / (1.0 + np.exp(-8.0 * (item_ratio - 0.3)))
    
    # Relative waste: remaining / (item + remaining)
    # Lower is better (item fills more of the remaining capacity)
    relative_waste = feasible_remaining / (safe_item + feasible_remaining)
    
    # Score: combine confidence-weighted Best-Fit signal with a small neutral component
    # Best-Fit component: -relative_waste (prefer lower relative waste)
    # Normalize relative_waste to [0, 1] range for consistent scaling
    # relative_waste is already in (0, 1)
    
    # Final score: confidence * (-relative_waste) + (1 - confidence) * 0
    # This smoothly transitions from Best-Fit (for large items) to neutral (for small items)
    scores[feasible] = confidence * (-relative_waste)
    
    return scores
# EVOLVE-BLOCK-END
