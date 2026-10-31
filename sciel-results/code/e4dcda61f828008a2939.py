"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import weibull_min

_START_TIME = time.time()
DEADLINE = 148.0
SAFETY_MARGIN = 2.0

# Pre-compute Weibull survival function parameters for the expected item distribution.
# Weibull shape k=2, scale lambda=20 (typical for bin packing problems with capacity 100).
# The survival function S(x) = exp(-(x/lambda)^k) gives the probability that a future item
# is larger than x. The expected waste of a remainder r is proportional to the integral
# of the survival function from r to infinity, which for Weibull is related to the
# upper incomplete gamma function. For simplicity, we approximate the "opportunity cost"
# as the survival function itself, S(r), since items larger than r cannot fit.
# Actually, the expected number of items that CAN fit is E[min(N, floor(capacity/r))] but that's complex.
# A simpler proxy: the value of a remainder r is the probability that a random future item
# is small enough to fit, i.e., 1 - S(r) = F(r) (the CDF).
# Wait, if we want to MINIMIZE waste, we want to avoid remainders that are likely to be wasted.
# A remainder r is wasted if no future item fits. The probability that a specific future item
# does NOT fit is S(r). If we assume a Poisson process of items, the probability that NO
# future item ever fits is more complex.
# Let's stick to the strategy: "Score each valid bin by the negative of this expected waste".
# Expected waste ~ S(r). So we want to MINIMIZE S(r).
# Minimizing S(r) means maximizing F(r) = 1 - S(r).
# This favors LARGER remainders (Worst Fit) because S(r) is smaller for larger r?
# No, S(r) is decreasing. S(0)=1, S(inf)=0.
# So minimizing S(r) means choosing the LARGEST r. This is Worst Fit.
# But Best Fit (smallest r) has S(r) close to 1 (high waste probability).
# This suggests Worst Fit is better under this model.
# However, the inspiration 2 (score 1.0204) used a hybrid.
# Let's look at the "opportunity cost" again.
# If we place an item in a bin, the remainder is r.
# The "cost" is the space that will remain empty.
# If r is small, it's likely to be filled by a small item.
# If r is large, it's likely to be filled by a large item, or wasted if items are small.
# The Weibull distribution with k=2, lambda=20 has mean ~ 17.7.
# Most items are < 30.
# If r > 30, it's very likely to be filled.
# If r < 5, it's unlikely to be filled (dead space).
# So we want to avoid r in the "dead zone" (small but not tiny) and also avoid r that is too large?
# Actually, Best Fit is known to be good for minimizing bins.
# Let's try a score that penalizes the "dead space" more heavily.
# Score = - (r * S(r)) ?
# Or simply use the CDF F(r) as the score? Higher F(r) means higher chance of being filled.
# F(r) = 1 - exp(-(r/lambda)^k).
# This is monotonic increasing. So it favors Worst Fit.
# But Best Fit is usually better. Why?
# Because Worst Fit opens more bins early on.
# Let's combine: Use Best Fit as the base, but adjust for the "value" of the remainder.
# If the remainder is in the "dead zone" (e.g., 5 < r < 15), it's bad.
# If r is very small (<5), it's "tight" and good (Best Fit).
# If r is large (>20), it's "flexible" and good (Worst Fit).
# So we want a U-shaped penalty? Or a score that is high for small r and large r?
# Let's try: Score = - |r - target| where target is something?
# Or use the derivative of the CDF?
# Let's go back to the inspiration 2's adaptive approach but with Weibull insights.
# The "dead zone" for Weibull(2, 20) is roughly (5, 15).
# Let's penalize remainders in this zone.
# Score = -r (Best Fit) + Penalty(r)
# Penalty(r) is high if r is in the dead zone.
# Penalty(r) = max(0, 1 - |r - 10|/5) ? A triangular bump centered at 10.
# Or simpler: just use the CDF F(r) as a "value" term and add it to the Best Fit score.
# Score = -r + alpha * F(r)
# If alpha is large, it becomes Worst Fit. If alpha is small, Best Fit.
# Let's try alpha = 0.1.

WEIBULL_K = 2.0
WEIBULL_LAMBDA = 20.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Weibull-Aware Adaptive Fit.
    Uses a blend of Best Fit and a Weibull-based "remainder value" score.
    """
    # Deadline check
    if time.time() > DEADLINE - SAFETY_MARGIN:
        feasible = bins >= item
        scores = np.full_like(bins, -np.inf, dtype=np.float64)
        if np.any(feasible):
            scores[feasible] = -(bins[feasible] - item)
        return scores

    # Identify feasible bins
    feasible = bins >= item
    if not np.any(feasible):
        return np.full_like(bins, -np.inf, dtype=np.float64)

    remaining = bins - item
    valid_remaining = remaining[feasible]

    # Calculate Weibull CDF for the remaining capacities
    # F(r) = 1 - exp(-(r/lambda)^k)
    # This represents the probability that a random item is small enough to fit in the remainder.
    # Higher F(r) means the space is more "useful".
    cdf_values = 1.0 - np.exp(-np.power(valid_remaining / WEIBULL_LAMBDA, WEIBULL_K))
    
    # Base score: Best Fit (minimize remaining)
    # score = -remaining
    # Adjustment: Reward useful remainders (high CDF)
    # This effectively shifts the preference towards larger remainders if they are "useful".
    # But since CDF is small for small r, it doesn't help Best Fit much for small r.
    # For large r, CDF is ~1, so it adds a constant.
    # This might not be enough to shift to Worst Fit.
    # Let's try a different adjustment: Penalize "dead" remainders.
    # Dead remainders are those with low probability of being filled, but not so small they are "tight".
    # Actually, small remainders ARE likely to be filled if items are small.
    # The issue is when r is larger than the typical item size.
    # If r > 30, F(r) is ~1.
    # If r = 10, F(10) = 1 - exp(-(10/20)^2) = 1 - exp(-0.25) = 1 - 0.778 = 0.222.
    # So a remainder of 10 has only 22% chance of being filled by a random item?
    # Wait, the CDF is the probability that an item is SMALLER than r.
    # So yes, only 22% of items are small enough to fit in a gap of 10.
    # This suggests that gaps of 10 are "wasteful" because 78% of items won't fit.
    # So we should penalize gaps where F(r) is low?
    # But Best Fit creates small gaps.
    # If r=2, F(2) = 1 - exp(-(2/20)^2) = 1 - exp(-0.01) = 0.01.
    # So only 1% of items fit in a gap of 2.
    # This implies Best Fit is BAD under this metric?
    # But Best Fit is empirically good. Why?
    # Because the items are not random in the future? Or because we want to fill the bin NOW.
    # The objective is to minimize the number of bins.
    # Using Best Fit ensures that the bin is "closed" as soon as possible.
    # If we leave a gap of 2, it's likely to stay empty, but the bin is still "open" and might take another item?
    # No, if the gap is 2, and the next item is 3, it doesn't fit. We open a new bin.
    # If we had used Worst Fit, we might have had a gap of 50, and the item 3 would fit.
    # So Worst Fit might be better for small items?
    # For large items, Best Fit is better.
    # Let's try a dynamic blend based on the item size.
    # If item is small, use Worst Fit (keep large gaps).
    # If item is large, use Best Fit (fill large gaps).
    
    # Let's define a weight w based on item size.
    # If item < 10, w = 1.0 (Worst Fit)
    # If item > 30, w = 0.0 (Best Fit)
    # Linear interpolation in between.
    if item < 10:
        w = 1.0
    elif item > 30:
        w = 0.0
    else:
        w = 1.0 - (item - 10) / 20.0
        
    # Score = w * (remaining) + (1-w) * (-remaining)
    # If w=1, Score = remaining (Worst Fit)
    # If w=0, Score = -remaining (Best Fit)
    
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    scores[feasible] = w * valid_remaining + (1 - w) * (-valid_remaining)
    
    return scores
# EVOLVE-BLOCK-END
