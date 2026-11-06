"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We implement a refined Best Fit heuristic.
# The goal is to minimize the number of bins used (excess over L1 lower bound).
# Best Fit (minimizing remaining capacity) is a strong baseline.
# We refine it by considering the "waste" created more aggressively.

# Key insight:
# If we place an item of size `item` in a bin with remaining capacity `r`,
# the new remaining capacity is `r' = r - item`.
# The "waste" or "gap" left in that bin is `r'`.
# We want to minimize the sum of gaps, or at least avoid creating large gaps.
# Simple Best Fit minimizes `r'`.
# However, sometimes minimizing `r'` is too greedy and leads to fragmentation
# where many bins have small, unusable gaps.
# But for Weibull distributions (which are often unimodal and skewed),
# Best Fit is known to perform very well.

# Let's look at the current implementation in the prompt's "CURRENT PROGRAM".
# It uses: `scores[valid] = -(remaining[valid]**2)`
# This is a quadratic penalty on the remaining capacity.
# This strongly prefers bins where the item fits very tightly (small `r'`).
# This is essentially "Best Fit" with a super-linear preference for tight fits.

# Is there a better function?
# Consider the "Waste" perspective.
# Total waste = Sum(remaining_i) for all bins.
# We want to minimize total waste.
# Locally, we want to minimize the new remaining capacity `r'`.
# The derivative of the total waste with respect to this choice is 1 (if we consider one bin).
# But we are replacing the previous state of that bin.
# Actually, the decision is: which bin to pick?
# If we pick bin `j`, the new remaining is `r_j - item`.
# The change in total waste is `(r_j - item) - r_j = -item`? No, that's not right.
# The total waste is the sum of remaining capacities of *all* bins.
# Before placing: Waste = Sum(r_i).
# After placing in bin j: Waste' = Sum_{i!=j} r_i + (r_j - item) = Waste - item.
# So the total waste decreases by `item` regardless of which bin we choose (as long as it fits).
# So minimizing total waste is not a differentiator in the immediate step?
# Wait, the number of bins is what matters for the score (bins_used).
# The score is `bins_used`.
# We open a new bin only if no existing bin has enough capacity.
# So, the goal is to **avoid opening new bins**.
# This means we should keep existing bins as "full" as possible so that future items can fit?
# No, if we keep them full (small remaining), they are less likely to fit future items.
# If we keep them with large remaining, they are more likely to fit future items.
# This suggests **Worst Fit** (maximizing remaining) might be better to keep options open?
# But Worst Fit is known to be bad for bin packing in general.
# Best Fit is generally better because it consolidates small items into the gaps of larger items,
# reducing the chance that a new bin is needed for a small item that would fit in a gap.

# Let's re-evaluate:
# If we use Best Fit, we fill the smallest gaps first.
# This tends to create bins that are "almost full".
# If a future item is large, it won't fit in these "almost full" bins, so we open a new bin.
# If we had used Worst Fit, we would have kept some bins with large gaps, which might fit the large item.
# However, Weibull distributions often have a mode and a tail.
# If the tail is light (small items), Best Fit is great.
# If the tail is heavy (large items), Worst Fit might be better?
# The problem states "five generated Weibull sequences". We don't know the shape parameter.
# Usually, standard benchmarks use shape parameters where Best Fit is competitive.

# Let's look at the "Quadratic" penalty again.
# `score = -(r')^2`.
# This heavily penalizes large `r'`.
# So it prefers the tightest fit.
# This is effectively "Best Fit" but with a stronger gradient.
# Is there a case where a slightly looser fit is better?
# Suppose Bin A has `r' = 1`, Bin B has `r' = 2`.
# Best Fit picks A.
# Next item is 1. It fits in A (if `r'=1` means remaining is 1, and item is 1, it fits exactly? No, `r'` is the remaining AFTER placing the current item.
# Let's clarify notation.
# `bins` array contains the **current remaining capacity** of each bin.
# `item` is the new item.
# `valid` bins have `bins[i] >= item`.
# `remaining_after = bins[i] - item`.
# We want to choose `i` to maximize the score.
# Current score: `-(bins[i] - item)^2`.
# This prefers the smallest `bins[i] - item`, i.e., the smallest `bins[i]` that is still >= `item`.
# This is exactly **Best Fit**.

# Why might it be stuck at 1.0?
# Maybe the seed program is already optimal for this specific distribution?
# Or maybe the evaluation is noisy?
# Or maybe there's a subtle improvement.

# Let's try a different heuristic: **First Fit Decreasing** is offline.
# Online equivalents:
# - Best Fit
# - Worst Fit
# - Next Fit
# - First Fit
# - Harmonic
# - Next Fit Decreasing (NFD) - but we don't have sorted items online.

# What about **Best Fit with a tie-breaker**?
# If two bins have the same `bins[i]`, it doesn't matter.

# What about considering the **distribution** of remaining capacities?
# If most bins have small remaining capacities, maybe we should be more aggressive?
# If most bins have large remaining capacities, maybe we should be more conservative (Worst Fit)?

# Let's try a **Hybrid** approach.
# Calculate the mean remaining capacity of valid bins.
# If the mean is low, use Best Fit (tight).
# If the mean is high, use Worst Fit (loose) to keep options open?
# Or maybe always Best Fit is better.

# Let's try to improve the Best Fit score slightly.
# Instead of `-(r')^2`, what if we use `-(r') * (1 + alpha * r')`?
# This is a generalized power penalty.
# For alpha > 0, it's super-linear.
# For alpha < 0, it's sub-linear.
# Current is effectively alpha = 1/r' (variable).

# Let's try **Worst Fit** for a moment.
# `score = bins[i]` (maximize remaining).
# This is known to be bad.

# Let's try **First Fit**.
# We don't have an order of bins? The `bins` array order might correspond to creation time?
# If so, First Fit would pick the first valid bin.
# `scores = np.where(valid, 1, -np.inf)`?
# This is usually worse than Best Fit.

# Let's stick to the Best Fit family.
# The current program uses `-(remaining)**2`.
# Let's try `-(remaining)**1.5`?
# Or `-(remaining)` (linear Best Fit)?
# The quadratic term might be too aggressive, leading to "over-packing" of small gaps,
# leaving larger bins unused which might be better for larger future items.
# However, if items are Weibull, they are often similar in size.
# If items are similar, Best Fit is very effective.

# Let's try a **smoothed** version or a **penalty** for fragmentation.
# Fragmentation is high if we have many bins with small, non-zero remaining capacities.
# We can't easily measure this without history.

# Let's try a simple change: **Linear Best Fit** vs **Quadratic Best Fit**.
# The prompt says the current program scores 1.0000.
# The current program in the prompt text is:
# `scores[valid] = -(remaining[valid]**2)`
# So the "seed" is Quadratic Best Fit.
# Maybe Linear Best Fit is better?
# `scores[valid] = -(remaining[valid])`
#
# Let's think about the difference.
# Case: Item = 10.
# Bin A: Remaining 11. New Rem 1.
# Bin B: Remaining 20. New Rem 10.
# Quadratic: A -> -1, B -> -100. Pick A.
# Linear: A -> -1, B -> -10. Pick A.
# Both pick A.
#
# Case: Item = 10.
# Bin A: Remaining 11. New Rem 1.
# Bin B: Remaining 12. New Rem 2.
# Quadratic: A -> -1, B -> -4. Pick A.
# Linear: A -> -1, B -> -2. Pick A.
# Both pick A.
#
# The ranking is the same for monotonic decreasing functions of `remaining`.
# `-(r')^2` and `-r'` both prefer smaller `r'`.
# So they make the **same choice** in all cases!
# The score values are different, but the argmax is the same.
# Therefore, changing the power of the penalty does **not** change the behavior,
# as long as the function is strictly decreasing in `remaining`.
#
# So, to improve, we must **change the preference** for which bin to pick.
# We need a non-monotonic function or a different metric.
#
# What metric is better than "minimize remaining after placement"?
#
# Idea: **Minimize the "waste" relative to the bin size?**
# No, capacity is constant.
#
# Idea: **Consider the "next" item?**
# We don't know it.
#
# Idea: **Worst Fit** (Maximize remaining after placement).
# `score = remaining[valid]`
# This prefers the bin with the **largest** remaining capacity that fits the item.
# This keeps the "big" bins open for large items.
#
# Let's compare Best Fit vs Worst Fit for Weibull.
# Weibull(1) is Exponential.
# Weibull(2) is Rayleigh.
# Weibull(3) is close to Normal.
# For Exponential, items vary widely.
# Best Fit tends to fill small gaps.
# Worst Fit tends to keep large gaps.
#
# Research suggests that for **unimodal** distributions, Best Fit is often superior to Worst Fit.
# However, there is a heuristic called **Best Fit Decreasing** which is offline.
# Online, **Best Fit** is standard.
#
# Is there a **Hybrid**?
# "Best Fit" if the item is small, "Worst Fit" if the item is large?
# If the item is large, it will only fit in bins with large remaining capacity.
# The choice among those bins:
# Best Fit would pick the one with the smallest remaining capacity (tightest fit).
# Worst Fit would pick the one with the largest remaining capacity.
#
# For a large item, the "gap" left behind is `r' = r - item`.
# If we pick the tightest fit, `r'` is small.
# If we pick the loosest fit, `r'` is large.
# A large `r'` is more useful for future items.
# So for large items, **Worst Fit** might be better.
# For small items, **Best Fit** is better because they can fit in many gaps, and we want to consolidate.
#
# Let's implement a **Hybrid Heuristic**:
# If `item` is small (e.g., < 25% of capacity), use Best Fit.
# If `item` is large (e.g., > 25% of capacity), use Worst Fit.
#
# Capacity is 100.
# Threshold: 25.
#
# Let's try this.
#
# `threshold = 0.25 * 100` (We don't know capacity exactly, but it's 100 in the task).
# Actually, we can infer capacity from the max value in `bins`?
# No, `bins` contains remaining capacities. The max possible remaining is 100 (if empty).
# But we don't know if a bin is empty or just has 100 remaining.
# We can assume capacity is the max value seen? Or just use a fixed threshold relative to the item?
#
# Let's use a threshold based on the item size relative to the **mean** valid remaining capacity?
# Or just a fixed threshold like 30?
#
# Let's try:
# If `item` < 30: Best Fit (minimize `r'`)
# If `item` >= 30: Worst Fit (maximize `r'`)
#
# Score for Best Fit: `-(r')`
# Score for Worst Fit: `r'`
#
# We need to combine them into a single score array.
#
# `r_prime = bins[valid] - item`
# `mask_small = item < 30`
#
# If `mask_small`:
#    `scores[valid] = -r_prime`
# Else:
#    `scores[valid] = r_prime`
#
# This is a discontinuous heuristic.
#
# Let's refine the threshold.
# Maybe 0.3 * capacity?
#
# Let's also consider that the "Worst Fit" for large items might be too aggressive.
# Maybe a "Soft Worst Fit"?
#
# Let's try this Hybrid approach.

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining capacity after placing the item
    remaining_after = bins - item
    
    # Identify valid bins (enough capacity)
    valid = remaining_after >= 0
    
    # Initialize scores to -inf
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    if not np.any(valid):
        return scores
    
    r_prime = remaining_after[valid]
    
    # Hybrid Heuristic
    # Capacity is 100.
    # Threshold for "large" items.
    # Let's try 30.
    threshold = 30.0
    
    if item < threshold:
        # Best Fit: minimize remaining_after
        # Score = -r_prime
        scores[valid] = -r_prime
    else:
        # Worst Fit: maximize remaining_after
        # Score = r_prime
        scores[valid] = r_prime
        
    return scores
# EVOLVE-BLOCK-END
