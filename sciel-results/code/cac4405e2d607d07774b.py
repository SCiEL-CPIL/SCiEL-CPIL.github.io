"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


# Weibull distribution parameters for the generated sequences
# Typical Weibull(100, k=2) has mean 100 * Gamma(1.5) / 2^0.5 ≈ 100 * 0.8862 / 1.4142 ≈ 62.67
# But the problem says "capacity 100" and items from Weibull. Let's assume scale=100, shape=2
# Mean = scale * Gamma(1 + 1/shape) = 100 * Gamma(1.5) ≈ 100 * 0.8862 = 88.62
# Median = scale * (ln(2))^(1/shape) = 100 * (0.6931)^0.5 ≈ 83.25
# Mode = scale * ((shape-1)/shape)^(1/shape) = 100 * (1/2)^0.5 ≈ 70.71

# The L1 lower bound for bin packing is sum(items) / capacity.
# Excess = (bins_used - lower_bound) / lower_bound * 100

# We want to minimize the mean percent excess.
# This is equivalent to minimizing bins_used / lower_bound, or just bins_used since lower_bound is fixed for a given sequence.
# So we want to minimize the number of bins used.

# Best-fit decreasing (BFD) is a classic heuristic.
# Best-fit: place item in the bin with the smallest remaining capacity that can still fit the item.
# This tends to consolidate items and reduce fragmentation.

# The current seed program uses: return -(bins - item)
# This means it prefers bins with smaller (bins - item), i.e., smaller remaining capacity after placing the item.
# This is exactly Best-Fit.

# Let's think about improvements.
# 1. First-Fit Decreasing (FFD) is not possible online.
# 2. Best-Fit Decreasing (BFD) is not possible online.
# 3. Online algorithms:
#    - Next-Fit: place in the first bin that fits.
#    - First-Fit: place in the first bin (in order) that fits.
#    - Best-Fit: place in the bin with the smallest remaining capacity that fits.
#    - Worst-Fit: place in the bin with the largest remaining capacity that fits.

# The seed is Best-Fit.
# Maybe a combination or a variant?
# "Probabilistic fragmentation penalty" was mentioned in the knowledge.
# "Discourages creating gaps too small for the expected next Weibull item."

# Let's try to incorporate the expected next item size.
# If we place an item in a bin, the remaining capacity is `rem`.
# If `rem` is too small to fit a typical next item, it's "dead space".
# We can penalize bins where the remaining capacity is close to the expected item size.

# Expected item size for Weibull(100, 2) is ~88.62.
# But items vary.
# Maybe use the mean or median.

# Let's try a scoring function:
# score = -remaining_capacity - penalty(remaining_capacity)
# where penalty is high if remaining_capacity is in a "dead zone" (e.g., < mean_item_size).

# Or, we can use a smooth approximation.
# Let's stick to a simple heuristic first.
# The knowledge says: "Replaces linear consolidation bias with a probabilistic fragmentation penalty... Measured: 0.9998 -> 1.042."
# This suggests that Best-Fit (linear consolidation) can be improved by a fragmentation penalty.

# Let's implement a Best-Fit with a fragmentation penalty.
# Penalty: if remaining_capacity < threshold, add a penalty.
# Threshold could be the mean item size.

# However, we don't know the exact distribution parameters.
# The problem says "five generated Weibull sequences of 1000 items with capacity 100".
# It doesn't specify the shape and scale.
# Usually, Weibull(100, 2) is a common choice.
# Let's assume mean_item_size = 88.62.

# But wait, if the mean is 88.62, then most items are large.
# A bin of capacity 100 can only hold 1 item on average.
# If we place an item of size 80, remaining is 20.
# The next item is likely > 20, so the 20 is dead space.
# Best-Fit would prefer this bin over a bin with remaining 50 (if 50 < 80, it's also dead space, but maybe less "dead" if the next item is small?).
# Actually, if the next item is likely large, we want to keep large chunks of space open.
# So maybe we should prefer bins with LARGER remaining capacity? (Worst-Fit?)
# But Worst-Fit is usually worse for bin packing.

# Let's reconsider.
# If items are large (mean 88), then a bin can usually only hold 1 item.
# So the number of bins is roughly the number of items.
# The lower bound is sum(items)/100.
# If items are ~88, sum(items) ~ 88 * 1000 = 88000.
# Lower bound = 880.
# Number of bins = 1000 (if each bin holds 1 item).
# Excess = (1000 - 880) / 880 * 100 = 120 / 880 * 100 = 13.6%.

# Can we do better?
# Only if we can fit 2 items in a bin.
# This requires item1 + item2 <= 100.
# Since mean is 88, this is rare.
# So the strategy should be to minimize the number of bins by fitting 2 items where possible.
# To fit 2 items, we need small items.
# Weibull(100, 2) has a long tail to the left.
# P(X < 50) = 1 - exp(-(50/100)^2) = 1 - exp(-0.25) = 1 - 0.7788 = 0.2212.
# So ~22% of items are < 50.
# If we have two small items, say 40 and 40, we can fit them in one bin.
# So we should try to group small items together.

# Best-Fit does this naturally: it places an item in the bin with the least remaining space.
# If we have a bin with 60 remaining, and an item of 40, Best-Fit will place it there (remaining 20).
# If we have a bin with 100 remaining, and an item of 40, Best-Fit will prefer the 60 bin.
# This consolidates small items.

# So Best-Fit is a good starting point.
# The fragmentation penalty might help by avoiding creating small gaps that can't be filled.
# But if we consolidate small items, we create small gaps (e.g., 20).
# If the next item is 30, it can't fit in the 20 gap.
# So the 20 is dead space.
# Maybe we should avoid creating gaps that are too small?
# If we place a 40 in a 100 bin, remaining is 60.
# If the next item is 50, it fits (remaining 10).
# If the next item is 70, it doesn't fit.
# So a gap of 60 is more useful than a gap of 20.

# So maybe we should prefer bins with LARGER remaining capacity?
# But that's Worst-Fit, which is usually bad.
# However, in this specific case (large items), maybe Worst-Fit is better?
# Let's test Worst-Fit: return (bins - item).
# This prefers bins with the most remaining space.
# This keeps large chunks of space open for large items.
# This might be better if items are large.

# Let's try a hybrid:
# If remaining capacity is > mean_item_size, prefer larger remaining capacity (Worst-Fit).
# If remaining capacity is < mean_item_size, prefer smaller remaining capacity (Best-Fit).
# This way, we keep large spaces open for large items, and consolidate small spaces for small items.

# Let's implement this.
# mean_item_size = 88.62 (approx).
# Let's use 80 as a threshold.

# score = -remaining_capacity if remaining_capacity < 80 else remaining_capacity
# This is a V-shaped function.

# Let's try this.

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    # Only consider bins with enough capacity
    valid = remaining >= 0
    # For invalid bins, assign a very low score
    scores = np.full_like(remaining, -np.inf, dtype=float)
    
    # Threshold for fragmentation penalty
    threshold = 80.0
    
    # For valid bins, calculate score
    valid_remaining = remaining[valid]
    
    # V-shaped score: prefer smaller remaining if < threshold, larger if > threshold
    # This encourages consolidating small gaps and keeping large gaps open.
    valid_scores = np.where(valid_remaining < threshold, -valid_remaining, valid_remaining)
    
    scores[valid] = valid_scores
    
    return scores

# EVOLVE-BLOCK-END
