"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

# Weibull parameters inferred from typical bin packing benchmarks
# Assuming shape (k) ~ 2 and scale (lambda) ~ 10-20 for items in [0, 100]
# We estimate the expected size of the next item to determine "safe" gap sizes.
WEIBULL_K = 2.0
WEIBULL_LAMBDA = 12.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Determine which bins have enough remaining capacity
    valid = bins >= item
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
        
    # Compute remaining capacity after placing the item
    remaining = bins - item
    
    # Initialize scores with a base preference for Best Fit (minimize remaining)
    # This ensures we don't waste space unnecessarily.
    scores = -remaining
    
    # Calculate the probability that the next item (from the Weibull distribution)
    # will fit into the remaining space of each bin.
    # P(next_item <= remaining) = 1 - exp(-(remaining/lambda)^k)
    # We want to avoid creating gaps that are too small for the next item.
    # If the probability of the next item fitting is low, the gap is likely "dead".
    
    # To avoid numerical issues with very small remaining values, clip to 0
    rem_clipped = np.maximum(remaining, 0.0)
    
    # Compute the CDF of the Weibull distribution at 'remaining'
    # This gives the probability that a random next item is smaller than 'remaining'
    # Note: If remaining is 0, prob is 0.
    # We use a small epsilon to avoid division by zero or log issues if we were taking logs, 
    # but here we just use the value directly.
    prob_fit = 1.0 - np.exp(-np.power(rem_clipped / WEIBULL_LAMBDA, WEIBULL_K))
    
    # Strategy:
    # We prefer bins where the remaining space is either:
    # 1. Very small (close to 0), so it's essentially filled (Best Fit).
    # 2. Large enough that the next item will likely fit (prob_fit is high).
    # We penalize bins where the remaining space is in the "dead zone":
    # Not small enough to be considered full, but not large enough for the next item.
    
    # Let's define a penalty for the "dead zone".
    # If prob_fit is low (e.g., < 0.5), the gap is likely useless.
    # We apply a penalty proportional to (1 - prob_fit) * remaining.
    # This discourages leaving medium-sized gaps that won't be used.
    
    # However, we also want to preserve large gaps for large items.
    # A better approach: Score = -remaining + bonus_if_likely_to_be_used.
    # Or: Score = -remaining * (1 + penalty_factor * (1 - prob_fit)).
    
    # Let's try a multiplicative penalty on the "waste".
    # If prob_fit is high, the waste is "potential usage", so we care less about it?
    # No, if prob_fit is high, the space *will* be used, so it's not waste yet.
    # If prob_fit is low, the space is likely waste, so we should avoid it.
    
    # Let's penalize the remaining space more heavily if it's unlikely to be used.
    # penalty = (1 - prob_fit) * remaining
    # score = -remaining - penalty = -remaining * (1 + (1 - prob_fit)) = -remaining * (2 - prob_fit)
    
    # This makes the penalty stronger for gaps where the next item won't fit.
    # If prob_fit = 1 (very large gap), score = -remaining * 1 (standard best fit).
    # If prob_fit = 0 (tiny gap), score = -remaining * 2 (double penalty, avoid tiny gaps).
    # This encourages either filling bins completely or leaving very large gaps.
    
    scores = -remaining * (2.0 - prob_fit)
    
    # Mask invalid bins
    scores[~valid] = -np.inf
    
    return scores
# EVOLVE-BLOCK-END
