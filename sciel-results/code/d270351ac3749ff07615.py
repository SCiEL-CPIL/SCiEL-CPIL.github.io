"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute constants for Weibull(1, 1.5) with capacity 100
# Weibull(1, 1.5) has mean = 1.5 * Gamma(1 + 1/1.5) = 1.5 * Gamma(1.6667)
# Gamma(1.6667) ≈ 0.9027, so mean ≈ 1.354
# For 1000 items, total size ≈ 1354, L1 lower bound ≈ ceil(1354/100) = 14 bins
# But we need a robust heuristic that works online.

# Key insight from the knowledge: opportunity_cost term penalizes remainders
# that are too small to be useful for most future items. This complements
# dead_space which penalizes medium remainders.

# Strategy: Use a composite score that combines:
# 1. Best-fit (minimize remaining space after placing item)
# 2. Opportunity cost: penalize bins where the remaining space after placing
#    the item is too small to fit a "typical" future item
# 3. A small bonus for bins that are nearly empty (to avoid fragmenting)

# Weibull(1, 1.5) item sizes: most items are small, with a long tail.
# Median of Weibull(1, 1.5) = (ln 2)^(1/1.5) ≈ 0.585
# Mean ≈ 1.354
# 75th percentile ≈ (ln 4)^(1/1.5) ≈ 1.26
# 90th percentile ≈ (ln 10)^(1/1.5) ≈ 2.07

# A "useful" remaining space should be at least ~2-3 units to accommodate
# a typical next item. Below that, it's "dead space."

# We use a smooth penalty: if remaining space r after placing item is small,
# we penalize it. The penalty is based on how unlikely it is that a future
# item will fit in that space.

# For Weibull(1, 1.5), the CDF is F(x) = 1 - exp(-x^1.5)
# So P(item <= r) = 1 - exp(-r^1.5)
# The probability that the NEXT item fits in remaining space r is F(r).
# If r is small, F(r) is small, meaning the space is "dead."

# We want to maximize the score, and the item goes to the highest-scoring bin.
# We want to PREFER bins that leave useful remaining space.

# Score = -best_fit_term + opportunity_bonus
# Where:
#   best_fit_term = remaining_space_after (lower is better, so negative)
#   opportunity_bonus = F(r) where r is remaining space after placing item
#     This rewards bins that leave space likely to be used by a future item.

# But we also want to avoid opening too many bins. A bin that is nearly full
# but leaves a tiny gap is worse than a bin that is moderately filled and
# leaves a useful gap.

# Let's define:
#   r = bin_capacity - bin_used - item  (remaining after placing)
#   score = -r + alpha * F(r)
# where F(r) = 1 - exp(-r^1.5) for Weibull(1, 1.5)

# But we need to be careful: if r is very large (bin is nearly empty), 
# F(r) approaches 1, which is good. If r is small, F(r) is small, which is bad.
# The -r term favors bins with less remaining space (best-fit).
# The +alpha*F(r) term favors bins with remaining space that is useful.

# These two terms work in tandem:
# - A bin with r=0.5: -0.5 + alpha*(1-exp(-0.35)) ≈ -0.5 + alpha*0.30
# - A bin with r=2.0: -2.0 + alpha*(1-exp(-2.83)) ≈ -2.0 + alpha*0.94
# - A bin with r=5.0: -5.0 + alpha*(1-exp(-11.2)) ≈ -5.0 + alpha*1.0

# With alpha=3:
# - r=0.5: -0.5 + 0.90 = 0.40
# - r=2.0: -2.0 + 2.82 = 0.82
# - r=5.0: -5.0 + 3.0 = -2.0

# This prefers r=2.0 over r=0.5 (good, useful gap) and over r=5.0 (good, 
# avoids wasting a nearly-empty bin).

# Let's tune alpha. The key is to balance best-fit (compactness) with 
# opportunity (useful remaining space).

# I'll use alpha = 2.5 as a starting point.

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score = preferred.
    
    bins: array of remaining capacities (or used amounts? Let's assume remaining)
    Actually, the seed program uses: return -(bins - item)
    which implies bins are remaining capacities, and bins - item is the 
    remaining after placing the item. So bins[i] >= item means the item fits.
    """
    # bins are remaining capacities
    # Filter bins that can fit the item
    mask = bins >= item
    
    if not np.any(mask):
        # No bin can fit the item; return all -inf or similar
        return np.full_like(bins, -np.inf, dtype=float)
    
    # Remaining space after placing the item
    r = bins - item  # This is >= 0 for valid bins
    
    # For bins that can't fit the item, set score to -inf
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Compute F(r) = 1 - exp(-r^1.5) for Weibull(1, 1.5)
    # Use a safe computation to avoid overflow/underflow
    r_safe = np.clip(r, 0, 100)
    r_15 = r_safe ** 1.5
    F_r = 1.0 - np.exp(-r_15)
    
    # Score = -r + alpha * F(r)
    alpha = 2.5
    valid_scores = -r + alpha * F_r
    
    # Only assign scores to bins that can fit the item
    scores[mask] = valid_scores[mask]
    
    return scores

# EVOLVE-BLOCK-END
