import numpy as np
import time

# Global state for deadline guard
_START_TIME = time.time()
_DEADLINE = 145.0  # 5 seconds margin before 150s limit

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the arriving item.
    The item goes to the highest-scoring bin.
    
    Strategy:
    - Model the packing process as a stochastic inventory system.
    - The cost of a remainder is the expected opportunity cost of holding that capacity.
    - Approximate this by the integral of the Weibull survival function from the remainder to infinity.
    - This quantifies the probability that the space remains unused by any future item.
    - Score each valid bin by the negative of this expected waste.
    - Combine with a smooth blend of Best-Fit and Worst-Fit to handle edge cases.
    """
    # Deadline check: if we are running out of time, fall back to simple Best Fit
    if time.time() > _DEADLINE:
        remaining = bins - item
        scores = np.full_like(bins, -1e9, dtype=float)
        valid = remaining >= 0
        scores[valid] = -remaining[valid]
        return scores

    # Calculate remaining capacity after placing the item
    remaining = bins - item
    
    # Mask for bins that can fit the item
    valid_mask = remaining >= 0
    
    # Initialize scores to a very low value (negative infinity) for invalid bins
    scores = np.full_like(bins, -1e9, dtype=float)
    
    # If no bin can fit the item, return all low scores
    if not np.any(valid_mask):
        return scores

    # Extract valid bins' remaining capacities
    valid_remaining = remaining[valid_mask]
    
    # Weibull parameters (assumed based on typical distributions in this task)
    # Shape parameter k and scale parameter lambda
    # Weibull survival function: S(x) = exp(-(x/lambda)^k)
    # Integral of S(x) from r to infinity = lambda * exp(-(r/lambda)^k) * gamma(1 - 1/k) / (r/lambda)^(k-1) * ...
    # Actually, the expected waste for a remainder r is the probability that no future item fits in r.
    # This is S(r) = exp(-(r/lambda)^k).
    # But we want the expected *unused* capacity, which is more complex.
    # Let's approximate the "opportunity cost" as S(r) * r, i.e., the probability the space is unused times the size of the space.
    # Or simply S(r) as a proxy for "waste probability".
    
    # Let's assume a Weibull distribution with k=0.8 and lambda=20 (typical for this kind of task)
    k = 0.8
    lam = 20.0
    
    # Weibull survival function: S(r) = exp(-(r/lam)^k)
    # For r=0, S(0)=1. For r=100, S(100)=exp(-(5)^0.8) ~ exp(-4.25) ~ 0.014
    # This means a remainder of 100 has a low probability of being used, which is good (low waste).
    # A remainder of 0 has a high probability of being "used" (or rather, no waste), so waste is low.
    # Wait, if r=0, there is no space to waste. So waste should be 0.
    # If r=100, there is a lot of space, but if it's unlikely to be used, is it waste?
    # Yes, if it's unlikely to be used, it's "dead" space, which is bad for bin packing efficiency.
    # So we want to minimize the expected dead space.
    # Dead space = r * S(r)
    
    # Let's compute expected waste = r * S(r)
    # S(r) = exp(-(r/lam)^k)
    # waste = r * exp(-(r/lam)^k)
    
    # We want to MINIMIZE waste, so we want to MAXIMIZE -waste.
    # So score = -waste = -r * exp(-(r/lam)^k)
    
    # However, this might not capture the full picture.
    # Let's combine with the previous Best-Fit/Worst-Fit blend.
    
    # Weight for Worst-Fit component
    # w=0 -> Best Fit, w=1 -> Worst Fit
    # Quadratic growth ensures w is small for most items
    w = (item / 100.0) ** 2
    
    # BestFitScore = -remaining
    # WorstFitScore = remaining
    # Blend: (1-w)*(-r) + w*(r) = r * (2w - 1)
    blend_score = valid_remaining * (2 * w - 1)
    
    # Stochastic waste score
    # waste = r * exp(-(r/lam)^k)
    # score_waste = -waste
    # But we need to normalize this to be comparable to blend_score.
    # blend_score ranges from -100 to 100.
    # waste ranges from 0 to ~100*exp(-1) ~ 37.
    # So they are on similar scales.
    
    # Let's compute the waste score
    # Handle r=0 case: waste = 0, score = 0
    # For r>0: score = -r * exp(-(r/lam)^k)
    
    # Use np.where to avoid log(0) or other issues, though exp handles 0 fine
    # (r/lam)^k for r=0 is 0, exp(0)=1, waste=0.
    
    r = valid_remaining
    waste = r * np.exp(-(r / lam) ** k)
    waste_score = -waste
    
    # Combine blend_score and waste_score
    # Let's use a weighted average.
    # alpha = 0.5 for equal weight
    alpha = 0.5
    final_scores = alpha * blend_score + (1 - alpha) * waste_score
    
    # Assign scores back to the original array
    scores[valid_mask] = final_scores
    
    return scores
