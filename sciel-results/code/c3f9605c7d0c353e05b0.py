import numpy as np
import time

# We need to keep the function name and signature.
# The task asks for a score where the item goes to the highest-scoring bin.
# The seed program uses Best Fit (minimize remaining capacity).
# The directive suggests investigating improvements within the Best Fit family,
# specifically considering a "fluid flow" perspective or smooth surrogates.
# However, the previous attempt was rejected. Let's look at the current program again.
# It implements Best Fit with a tiny index-based tie-breaker.

# One common improvement for Best Fit in online bin packing is to consider
# the "fragmentation" or the utility of the space left.
# Another heuristic is "Best Fit Decreasing" but we are online, so we don't know future items.
#
# Let's try a variant that slightly penalizes bins that are already very full,
# or conversely, slightly prefers bins that are not completely empty if we are in a "spread" phase?
# No, Best Fit is generally strong for minimizing the number of bins.
#
# The directive mentions: "Treat the packing process as a continuous fluid flow problem...
# prioritizing bins where the item constitutes a large fraction of the remaining capacity"
# This sounds like it wants to maximize the ratio item/remaining?
# If item is large relative to remaining, it fills it up well.
# If item is small relative to remaining, it leaves a lot of space.
# Best Fit minimizes remaining, which implies maximizing item/remaining?
# Not exactly. If remaining is small, item/remaining is large (if item > 0).
# So Best Fit already prioritizes bins where the item is a large fraction of the remaining capacity.
#
# Let's look at the "smooth surrogate" hint.
# Instead of a hard min (which is what argmin remaining does, effectively),
# maybe we can use a soft-min? But we just need to return scores.
# The evaluator picks the max score.
# If we return -remaining, the max score is the min remaining. That's Best Fit.
#
# What if we adjust the score to be a non-linear function of remaining?
# For example, score = -1 / (remaining + epsilon)?
# This would heavily penalize bins with large remaining capacity, pushing us towards Best Fit even more aggressively?
# Or score = -remaining^2? This also pushes towards Best Fit.
#
# Let's consider the "Worst Fit" alternative mentioned in the previous thought process.
# Worst Fit maximizes remaining. This is good for keeping space for large future items.
# For Weibull distributions, if the shape parameter is < 1, items are heavy-tailed (many small, few huge).
# Worst Fit might be better for heavy-tailed distributions to preserve large blocks of space.
# If shape > 1, items are more uniform or light-tailed. Best Fit is often better.
# The task says "five generated Weibull sequences". We don't know the shape parameters.
# Standard Weibull in bin packing benchmarks often uses shape 1.5 or 2.0 (light-tailed) or 0.5-0.8 (heavy-tailed).
#
# Let's try a hybrid or a specific tuning.
# A common strong heuristic is "Best Fit" with a slight modification.
#
# Let's try "Best Fit" but with a tie-breaker that prefers bins with *lower* remaining capacity more strongly?
# The current tie-breaker is just index.
#
# Actually, let's look at the "fluid flow" hint again.
# "minimize the relative 'waste' of opening new bins"
# This phrasing is tricky. Opening a new bin is "waste" in terms of bin count.
# So we want to minimize the number of bins. Best Fit is known to perform well for this.
#
# Let's try a slight variation: **Best Fit with a penalty for very small remainders?**
# No, small remainders are good (less waste).
#
# What if we use **First Fit Decreasing** logic? We can't sort.
#
# Let's try **Worst Fit** again.
# In many online bin packing scenarios with unknown item distributions, Worst Fit can outperform Best Fit if the distribution has a heavy tail (large items).
# If the Weibull shape is small (e.g., 0.5), there are large items. Worst Fit keeps large holes open.
# If the shape is large (e.g., 2.0), items are smaller and more uniform. Best Fit packs them tighter.
#
# Since we don't know the shape, maybe we can detect it? No, we only see one item at a time.
#
# Let's try a score that balances Best and Worst Fit.
# Score = -remaining + alpha * (item / remaining)?
#
# Let's stick to the family of Best Fit but refine the tie-breaking or the function shape.
# The current function: `scores[valid] = -remaining[valid]`
#
# Let's try `scores[valid] = -np.log(remaining[valid] + 1e-6)`
# This compresses the range of large remainders and expands the range of small remainders.
# This makes the difference between a bin with 1 unit remaining and a bin with 2 units remaining more significant than between 99 and 100.
# This is a "soft" Best Fit.
#
# Let's try this. It is within the "Best Fit" family (minimizing remaining) but with a non-linear scaling.

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining capacity
    remaining = bins - item
    
    # Identify valid bins (enough capacity)
    valid = remaining >= 0
    
    # Initialize scores with -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid):
        return scores
        
    # Non-linear Best Fit: 
    # Use -log(remaining + epsilon) to prioritize bins with very small remaining capacity more strongly.
    # This helps in packing small items into the "cracks" left by larger items.
    eps = 1e-6
    scores[valid] = -np.log(remaining[valid] + eps)
    
    # Tie-breaking: Prefer lower index (First Fit style)
    # Subtract a tiny penalty proportional to index
    scores[valid] -= np.arange(len(bins))[valid] * 1e-9
    
    return scores
