"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard
_START_TIME = time.time()
_DEADLINE = 145.0  # 5 seconds margin before 150s

# Precompute Weibull CDF lookup for gap utility estimation
# Weibull with shape k~1.5, scale lambda~12 (typical for capacity-100 bin packing)
# P(gap >= x) = 1 - F(x) where F is Weibull CDF
# We'll use a piecewise-linear approximation of the survival function S(x) = 1 - F(x)
# S(x) = exp(-(x/lambda)^k) for Weibull
_WEIBULL_K = 1.5
_WEIBULL_LAMBDA = 12.0

def _weibull_survival(x):
    """Compute P(X >= x) for Weibull(k, lambda), clipped to [0, 100]."""
    x = np.clip(x, 0, 100)
    return np.exp(-((x / _WEIBULL_LAMBDA) ** _WEIBULL_K))

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin. The item goes to the bin with the highest score.
    bins: array of remaining capacities for each bin.
    item: the size of the item to place.
    
    Strategy: Gap-Utility Scoring
    For each valid bin, compute the resulting gap (remaining capacity after placing item).
    Score the gap by its "fillability": the probability that a future Weibull-distributed
    item will fit in that gap. Bins that leave gaps most likely to be filled are preferred.
    This is a structural departure from deviation-based scoring: it directly models
    the expected future utility of the created gap using the item size distribution.
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

    # Extract valid bins' remaining capacities (gaps after placement)
    valid_remaining = remaining[valid_mask]
    
    # Core scoring: gap utility via Weibull survival function
    # A gap of size g has utility = P(future item fits in gap) = P(item <= g) = F(g)
    # But we also want to reward gaps that are "useful" - not too small (waste) 
    # and not too large (wastes bin capacity).
    # 
    # Key insight: The ideal gap is one that can be filled by a typical future item.
    # For Weibull(k=1.5, lambda=12), the median is ~12*(ln2)^(1/1.5) ≈ 12*0.69 ≈ 8.3
    # The mean is ~12*Gamma(1+1/1.5) ≈ 12*1.06 ≈ 12.7
    #
    # We score each gap by:
    #   1. Fillability: P(item_size <= gap) = F(gap) — higher is better
    #   2. Efficiency: penalize gaps much larger than needed (wastes capacity)
    #   3. Anti-fragmentation: penalize very small gaps that are hard to fill
    #
    # Combined: score = F(gap) * efficiency_factor - small_gap_penalty
    
    # Fillability: probability that a future Weibull item fits in this gap
    # F(g) = 1 - exp(-(g/lambda)^k)
    g = np.clip(valid_remaining, 0, 100)
    fillability = 1.0 - np.exp(-((g / _WEIBULL_LAMBDA) ** _WEIBULL_K))
    
    # Efficiency: prefer gaps close to the "sweet spot" where fillability is high
    # but the gap isn't excessively large. Use a Gaussian-like kernel centered
    # at the Weibull mean (~12.7) with moderate width.
    sweet_spot = _WEIBULL_LAMBDA * (1.0 + 1.0/_WEIBULL_K)  # approx mean of Weibull
    sigma_eff = 8.0
    efficiency = np.exp(-((g - sweet_spot) ** 2) / (2.0 * sigma_eff ** 2))
    
    # Anti-fragmentation: penalize gaps < 3 (too small to be useful for most items)
    # Use a smooth ramp: penalty is 1 for g<1, 0 for g>5
    frag_threshold_low = 1.0
    frag_threshold_high = 5.0
    frag_penalty = np.clip((frag_threshold_high - g) / (frag_threshold_high - frag_threshold_low), 0, 1)
    
    # Combine: primary signal is fillability weighted by efficiency
    # Subtract fragmentation penalty for tiny gaps
    scores_valid = fillability * (0.5 + 0.5 * efficiency) - 0.3 * frag_penalty
    
    # Add a small Best-Fit component to break ties and prefer tighter fits
    # when fillability is similar
    bestfit_component = -valid_remaining * 0.01
    
    scores_valid += bestfit_component
    
    scores[valid_mask] = scores_valid
    
    return scores
# EVOLVE-BLOCK-END
