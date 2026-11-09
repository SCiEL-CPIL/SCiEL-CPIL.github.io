"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the item.
    The item goes to the bin with the highest score.
    
    Strategy: First-Fit Decreasing-like heuristic adapted for online setting.
    We want to pack items tightly to minimize the number of bins used.
    A good heuristic is to prefer bins where the item fits 'best' in terms of
    leaving the least wasted space, but also considering future flexibility.
    
    Here we use a weighted combination:
    - Primary: minimize remaining capacity after placing item (best-fit)
    - Secondary: prefer bins that are already 'active' (have some items) to
      consolidate items, reducing the total number of bins opened.
    
    We score bins as:
    score = -remaining_capacity_after + lambda * (1 if bin is not empty else 0)
    
    Actually, a simpler and often effective approach for online bin packing
    is Best-Fit: place item in the bin with the smallest remaining capacity
    that can still fit the item. This tends to fill bins completely.
    
    However, pure best-fit can be suboptimal. Let's try a variant that
    considers the 'fill ratio' of the bin.
    
    Let bins be the array of current loads (or remaining capacities?).
    Looking at the seed: `return -(bins - item)`.
    If `bins` represents remaining capacity, then `bins - item` is the new
    remaining capacity after placing the item. Minimizing this (i.e., making
    it as small as possible) is Best-Fit.
    
    Let's try a different approach: Next-Fit Decreasing is offline.
    For online, common heuristics are:
    1. First Fit: first bin that fits.
    2. Best Fit: bin with least remaining space that fits.
    3. Worst Fit: bin with most remaining space that fits.
    
    The seed uses Best Fit (by negating the remaining space, so higher score
    = less remaining space).
    
    Let's try Worst Fit: prefer bins with the most remaining capacity.
    This might help by keeping bins 'open' with more room for future large items.
    
    Or, a hybrid: if the item is large, use Best Fit; if small, use Worst Fit?
    
    Let's try: score = remaining_capacity_after. We want to maximize this?
    No, the contract says "item goes to the highest-scoring bin".
    
    If we want Best Fit: we want to minimize `bins - item`. So score = `-(bins - item)`.
    If we want Worst Fit: we want to maximize `bins - item`. So score = `bins - item`.
    
    Let's try Worst Fit (score = bins - item) to see if it performs better.
    Worst Fit tends to spread items out, which might be bad for minimizing bin count.
    
    Another idea: Use a priority based on the bin's current load.
    Prefer bins that are already heavily loaded (to consolidate).
    score = bins (if bins is load) or score = -(bins) if bins is remaining.
    
    Let's assume `bins` is the remaining capacity.
    Load = capacity - bins.
    
    Best Fit: minimize remaining after placement -> score = -(bins - item)
    Worst Fit: maximize remaining after placement -> score = (bins - item)
    
    Let's try a 'First Fit' like behavior but with a twist.
    Actually, let's look at the L1 lower bound. The L1 lower bound is sum(items) / capacity.
    To get close to this, we need to pack efficiently.
    
    A known good heuristic is 'Best Fit Decreasing' but we are online.
    
    Let's try: score = bins - item (Worst Fit).
    Why? By keeping bins with lots of space, we might accommodate future large items
    that wouldn't fit in a nearly full bin. This can reduce the number of bins
    needed if item sizes are variable.
    
    Let's implement Worst Fit.
    """
    remaining = bins - item
    # Only bins with enough capacity are valid.
    # The problem states: "scores each bin with enough remaining capacity"
    # and "the item goes to the highest-scoring bin".
    # We should return -inf or very low score for bins that can't fit the item.
    
    # Create a mask for valid bins
    valid = bins >= item
    
    # For Worst Fit, we want to maximize remaining space after placement.
    scores = np.full_like(bins, -np.inf, dtype=float)
    scores[valid] = remaining[valid]
    
    return scores
# EVOLVE-BLOCK-END
