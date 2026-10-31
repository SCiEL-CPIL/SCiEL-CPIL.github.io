"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


# We cache some state to enable a slightly smarter heuristic that looks at
# the distribution of remaining capacities, not just the immediate fit.
# However, since the function is stateless (bins is passed in), we must compute
# scores based solely on the current state of `bins` and `item`.

# The seed program uses Best-Fit (minimize remaining capacity).
# A common improvement for bin packing is "Best Fit Decreasing" but we don't see future items.
# Another approach is "Worst Fit" (maximize remaining capacity) which can sometimes help
# by keeping large chunks free for large future items, but usually performs worse than Best Fit
# for uniform distributions. For Weibull, which has a shape parameter, the distribution might
# be skewed.

# Let's try a heuristic that balances Best Fit with a penalty for fragmentation.
# Or, simply try "First Fit" equivalent? No, we score all bins.
# 
# Let's analyze the "Excess over L1 lower bound".
# L1 lower bound is usually max(sum(items), max(item)).
# Minimizing the number of bins is the primary goal.
#
# A sophisticated heuristic:
# Score = - (remaining_capacity) + alpha * (some measure of bin "health")
#
# Actually, a very effective online algorithm is "Best Fit with a slight preference for bins that are already partially filled" 
# which is essentially Best Fit. 
#
# Let's try "Worst Fit" to see if it helps with Weibull distributions where large items might arrive later.
# Or "First Fit" which is equivalent to picking the first available bin? No, we pick the highest score.
#
# Let's try a hybrid: 
# If the item fits in a bin with very little left, it's a great fit (Best Fit).
# If the item fits in a bin with a lot left, it might be good to save that space (Worst Fit).
#
# A known good heuristic for online bin packing is "Best Fit". 
# Another is "Harmonic" or "Slack" heuristics.
#
# Let's try scoring based on the *relative* remaining capacity compared to the item size.
# Or simply: return -(bins - item) is Best Fit.
#
# What if we use a non-linear penalty?
# score = -np.log1p(bins - item) ?
# This favors bins where the remaining capacity is small, but less aggressively than linear.
#
# Let's try "Worst Fit" first as a structural change from Best Fit.
# score = bins - item (highest remaining capacity)
#
# Wait, the prompt says "Fresh Design". 
# Let's look at the distribution. Weibull with shape k.
# If k > 1, it's right-skewed (many small, few large).
# If k < 1, it's left-skewed (many large, few small) - actually Weibull is always non-negative.
# Standard Weibull: f(x) = (k/lambda)(x/lambda)^(k-1) exp(-(x/lambda)^k).
# Mean is lambda * Gamma(1 + 1/k).
#
# If the distribution has many small items, Best Fit is usually good.
# If there are large items, Worst Fit might be better to keep big bins open.
#
# Let's try a "Hybrid Fit" that switches based on the item size relative to the bin capacity (100).
# If item is small, use Best Fit. If item is large, use Worst Fit?
#
# Let's try: score = -(bins - item) ** 0.5  (Root Best Fit)
# Or: score = -(bins - item) / (bins + 1)
#
# Let's try a simple structural change: **Worst Fit**.
# This is a fundamentally different strategy than Best Fit.
#
# Alternative: **First Fit** is not directly implementable as a scoring function over all bins unless we simulate order, but the bins array doesn't preserve "order" of creation necessarily, or does it? 
# The problem says "bins: np.ndarray". It's likely an array of current remaining capacities.
# If the bins are just a set of capacities, "First Fit" isn't well-defined unless the array index corresponds to creation time.
# Assuming the array index corresponds to the order bins were opened, First Fit would be:
# Find the first bin (lowest index) that can fit the item, and give it a huge score, others -inf.
# But we must return a score for *each* bin with enough capacity.
# We can set the first valid bin to a very high score and others to lower scores.
#
# Let's try **First Fit** heuristic.
# Logic:
# 1. Identify bins where bins >= item.
# 2. Among those, find the one with the smallest index (assuming bins[0] is the oldest/first bin).
# 3. Assign a very high score to that bin.
# 4. Assign lower scores to others to ensure the first one is picked.
#
# This is a structural change from "Best Fit" (minimize residual) to "First Fit" (minimize index).
# First Fit often performs well because it tends to fill up early bins completely, leading to fewer partially filled bins at the end.
#
# Implementation:
# valid_mask = bins >= item
# if no valid, return zeros or -inf? The task says "enough remaining capacity".
# We should return -inf for invalid bins so they are never picked.
#
# For First Fit:
# indices = np.where(valid_mask)[0]
# if len(indices) == 0: return np.full_like(bins, -np.inf)
# first_idx = indices[0]
# scores = np.full_like(bins, -1.0)  # Low score for valid bins
# scores[first_idx] = 1.0             # High score for the first valid bin
# scores[~valid_mask] = -np.inf       # Invalid bins
# return scores

# Let's try this.

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    First Fit heuristic: Prefer the earliest opened bin that can fit the item.
    This encourages filling bins to capacity before opening new ones or using later bins.
    """
    # Ensure bins is a numpy array
    bins = np.asarray(bins, dtype=float)
    
    # Initialize scores with -inf for all bins
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Find bins that can fit the item
    valid_mask = bins >= item
    
    if not np.any(valid_mask):
        # No bin can fit the item. The caller should handle this (open a new bin).
        # Return all -inf.
        return scores
    
    # Get indices of valid bins
    valid_indices = np.where(valid_mask)[0]
    
    # First Fit: pick the valid bin with the smallest index
    first_idx = valid_indices[0]
    
    # Assign high score to the first valid bin
    # Assign slightly lower scores to other valid bins to break ties if the first one is somehow rejected (though it shouldn't be)
    # Actually, just setting the first one to 1.0 and others to 0.0 is sufficient.
    scores[valid_mask] = 0.0
    scores[first_idx] = 1.0
    
    return scores
# EVOLVE-BLOCK-END
