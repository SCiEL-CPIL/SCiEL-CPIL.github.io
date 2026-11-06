"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We track a simple heuristic: Best Fit Decreasing is strong for online bin packing
# when we can't see the future. The seed is pure Best Fit.
# Let's try a variant that combines Best Fit with a slight preference for bins
# that are not too empty (to avoid fragmentation) but also not too full.
# Actually, let's try "Best Fit" but with a tie-breaking or slight modification.
# Or try "Worst Fit" which is sometimes competitive.
# Or try a hybrid: prefer bins where the remaining space after placing the item
# is "useful" for future items. Since items are Weibull, we might guess the distribution.
# However, the function is stateless per call (except for the bins array which reflects past placements).

# Let's try a different approach: 
# The L1 lower bound is sum(items) / capacity.
# To minimize the number of bins used, we want to pack as tightly as possible.
# Best Fit is known to be very strong.
# Maybe we can improve it by considering the "waste" created.
# When we place an item of size 'item' in a bin with remaining 'r', the new waste is 'r - item'.
# Best Fit minimizes this waste.

# Let's try a score that penalizes large bins more, to encourage filling smaller bins first?
# No, Best Fit already does that implicitly if bins are homogeneous in capacity.
# The bins array contains the remaining capacity. So all bins start at 100.
# As items are placed, bins shrink.

# Let's try a simple modification: 
# score = -(bins - item)  # Best Fit
# What if we add a small bonus for bins that are "half full"?
# Or what if we use a non-linear penalty?

# Let's try: score = -((bins - item) ** 0.5)
# This makes the difference in waste less important for large waste, and more important for small waste?
# No, sqrt is concave, so it penalizes small waste more? 
# d/dx sqrt(x) = 1/(2 sqrt(x)). As x -> 0, derivative -> infinity. So it strongly penalizes very small waste?
# Wait, we want to MINIMIZE waste. So we want to MAXIMIZE score.
# If score = -sqrt(remaining - item), then minimizing remaining-item is still the goal, but the "cost" of waste is sublinear.
# Actually, let's just stick to a robust heuristic.

# Another idea: First Fit Decreasing is not possible online.
# Best Fit is the standard online heuristic.

# Let's try "Worst Fit" as a contrast.
# Worst Fit tends to keep bins balanced, which can be good for Weibull distributions with many small items.

# Let's implement a hybrid or a different heuristic.
# How about: score = - (bins - item) / (bins + item)
# This normalizes the waste relative to the bin size.

# Let's try a simple alternative: 
# Prefer the bin that leaves the remaining capacity closest to the item size?
# No, that's complex.

# Let's try "Best Fit" with a slight preference for the bin that has the *largest* remaining capacity among those with minimal waste?
# No, that would be Worst Fit among the best fits.

# Let's try: 
# score = - (bins - item) + 0.1 * bins
# This is a linear combination of Best Fit and Worst Fit.
# If weight on Worst Fit is small, it's mostly Best Fit.

# Let's try: 
# score = - (bins - item) ** 2
# Quadratic penalty on waste. This makes Best Fit even more strict about small waste differences.

# Let's implement Quadratic Best Fit.
# It might be more sensitive to small differences in remaining space, leading to tighter packing.

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid = remaining >= 0
    scores = np.full_like(bins, -np.inf, dtype=float)
    if np.any(valid):
        # Quadratic penalty: heavily favor bins with least waste
        scores[valid] = - (remaining[valid] ** 2)
    return scores
# EVOLVE-BLOCK-END
