import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid = remaining >= 0

    # Novel approach: "Balanced Fill" heuristic.
    # Instead of purely minimizing (Best-Fit) or maximizing (Worst-Fit) remaining space,
    # we score bins based on how well the item fills the bin relative to a "sweet spot".
    # 
    # Intuition:
    # - Best-Fit (score = -remaining) tends to pack tightly, which is good for small items
    #   but can lead to fragmentation where small gaps are left that can't be filled.
    # - Worst-Fit (score = remaining) spreads items out, which is good for large items
    #   to save space for other large items, but can waste space if items are small.
    #
    # This heuristic uses a quadratic penalty on the remaining space.
    # score = - (remaining / capacity)^2 * capacity
    # This effectively penalizes large remaining spaces more heavily than linearly,
    # but less aggressively than Best-Fit for very small remaining spaces.
    # It creates a "soft" Best-Fit that is less prone to over-packing small items
    # into tiny gaps, potentially improving performance on Weibull distributions
    # which have a tail of larger items.
    
    capacity = 100.0  # Assumed capacity from problem description
    
    # Normalize remaining space
    norm_remaining = remaining / capacity
    
    # Quadratic penalty: larger remaining space gets a much lower score
    # We want highest score to win, so we negate the penalty.
    # score = - (norm_remaining)^2 * capacity
    # This is equivalent to - (remaining^2) / capacity
    
    scores = np.where(valid, -(remaining ** 2) / capacity, -np.inf)
    
    return scores
