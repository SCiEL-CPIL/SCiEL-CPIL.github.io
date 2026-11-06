"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We can't easily do heavy computation inside priority per-call if it's called many times.
# The function signature is priority(item: float, bins: np.ndarray) -> np.ndarray.
# 'bins' is an array of remaining capacities.
# We need to return an array of scores, one per bin.
# The item goes to the highest-scoring bin.
# Bins with insufficient capacity should probably have a very low score (or -inf).

# Strategy:
# The L1 lower bound on the number of bins is ceil(sum(items) / capacity).
# Minimizing the number of bins is equivalent to maximizing the total packed volume,
# or minimizing the total wasted space.
# Online, we don't know future items.
# Best-fit decreases remaining capacity the most (leaves least slack).
# Worst-fit increases remaining capacity the most (leaves most slack).
# 
# Research suggests that for random items, Best-Fit Decreasing (if we could sort) is good, but we are online.
# Among online heuristics:
# - First Fit: First bin that fits.
# - Best Fit: Bin with least remaining space after placing item.
# - Worst Fit: Bin with most remaining space after placing item.
# - Next Fit: Only use current bin, open new if doesn't fit.
#
# The current program uses Best Fit: return -(bins - item).
# If bins[i] - item is small (tight fit), the score is high (less negative).
# If bins[i] - item is large (loose fit), the score is low (more negative).
# So it picks the tightest fit.
#
# Let's analyze "Worst Fit": return (bins - item).
# This picks the loosest fit. This tends to keep bins balanced, which can be good for avoiding fragmentation where small items don't fit anywhere.
#
# Let's analyze a hybrid or a different metric.
# "Relative Best Fit": minimize (bins - item) / bins?
# Or maximize the ratio of item to bin?
#
# Let's look at the score metric: "mean percent excess of bins used over the L1 lower bound".
# L1 bound = sum(items) / C.
# Excess = (BinsUsed - L1) / L1.
# We want to minimize BinsUsed.
#
# Best Fit is generally very strong for random distributions.
# Worst Fit is often competitive too.
#
# Is there a better heuristic?
# "Best Fit Decreasing" is offline.
# "Harmonic" or "Dual Fit" are more complex.
#
# Let's try a slight variation of Best Fit that penalizes very small remaining spaces? No, that would move away from Best Fit.
#
# What if we consider the *relative* waste?
# Waste = bins - item.
# Best Fit minimizes absolute waste.
#
# Let's try **Worst Fit**.
# In many online packing scenarios, Worst Fit performs similarly to or slightly better than Best Fit for uniform distributions, but for skewed distributions (like Weibull with shape < 1, which is high density near 0, or shape > 1), it varies.
# Weibull with shape k:
# If k=1 (Exponential), items are small on average.
# If k>1, items are more uniform/larger.
# The problem statement says "five generated Weibull sequences". It doesn't specify k.
# Usually, these benchmarks use k=1 or k=2.
#
# Let's consider the distribution of remaining capacities.
# If we use Best Fit, we fill bins up tightly. This creates bins with very little space.
# If we use Worst Fit, we spread items out, keeping bins half-full on average.
#
# There is a known result that **Worst Fit** can be better than Best Fit for certain item size distributions, particularly when items are large relative to capacity, or to maintain flexibility.
# However, Best Fit is the standard strong baseline.
#
# Let's look at the "percent excess".
# If we use fewer bins, we win.
#
# Let's try a **Next Fit** variant? No, that's usually worse.
#
# Let's try **First Fit**? No, Best Fit is generally better than First Fit.
#
# How about a combination?
# Score = -(bins - item) + alpha * (something else)?
#
# Let's look at the "L1 lower bound".
# L1 = sum(items) / C.
# To minimize bins, we want to pack as much as possible.
#
# Let's try **Worst Fit** (maximize remaining space).
# Code: return (bins - item)
#
# Let's try **Best Fit** but with a tie-breaker?
#
# Actually, there is a heuristic called **Best Fit Decreasing** which is optimal for small N, but we are online.
#
# Let's consider the specific nature of Weibull.
# Weibull(1) is Exponential.
# Weibull(2) is Rayleigh.
#
# In the paper "Online Bin Packing" or similar, **Worst Fit** is often cited as having a competitive ratio similar to Best Fit, but sometimes better in practice for specific distributions.
#
# Let's try a **Weighted Best Fit**.
# Score = -(bins - item) / bins?
# This favors bins that are already partially filled?
# If bin is 100, item is 10. Waste 90. Ratio 0.9.
# If bin is 20, item is 10. Waste 10. Ratio 0.5.
# Best Fit picks the 20 bin (Waste 10).
# Relative Best Fit picks the 20 bin (Ratio 0.5 is "better" if we minimize waste ratio? No, usually we want to fill up).
#
# Let's stick to the most robust heuristics.
#
# Alternative: **Harmonic** heuristic?
# It classifies items into classes based on size and puts them in bins of corresponding size.
# This is complex to implement online without knowing the item size distribution or pre-allocating bin sizes.
#
# Let's try **Worst Fit**.
# Why? Best Fit can lead to "fragmentation" where many bins have small gaps that no future item fits into. Worst Fit keeps gaps large, allowing larger future items to fit. Since Weibull distributions can have a wide range of item sizes (especially if k is small), keeping large gaps might be beneficial.
#
# Let's test the logic:
# Best Fit: `-(bins - item)`
# Worst Fit: `bins - item`
#
# Let's also consider **Next Fit** is bad.
# **First Fit** is bad.
#
# What about **Best Fit** with a penalty for creating *too small* a gap?
# If `bins - item` is very small, it's good.
#
# Let's try a different angle.
# The score is `mean percent excess`.
#
# I will switch to **Worst Fit** as a fresh design. It is a strong competitor to Best Fit and often outperforms it in random online settings due to better space preservation for larger items.
#
# Wait, is there a simpler improvement?
# The current code is Best Fit.
#
# Let's look at the deadline guard.
# The function is called per item. 1000 items.
# The computation must be very fast.
# `bins - item` is O(N) where N is number of bins.
# Number of bins is roughly sum(items)/C.
# For 1000 items of mean size ~10-20, sum ~10000-20000. C=100. Bins ~100-200.
# So N is small. Vectorized operations are fine.
#
# Let's implement Worst Fit.
#
# One more idea: **Best Fit** is minimizing waste.
# **Worst Fit** is maximizing remaining space.
#
# Let's try **Best Fit** but inverted? No.
#
# Let's try a **Hybrid**:
# If the item is small, use Worst Fit (to keep space).
# If the item is large, use Best Fit (to fit it).
#
# Threshold: 0.5 * Capacity?
# Capacity is not passed to `priority`. We only have `bins` (remaining capacities) and `item`.
# We can infer capacity? No, bins start at 100, but we don't know the original capacity from the function signature.
# Wait, the problem says "capacity 100". We can hardcode C=100?
# "five generated Weibull sequences of 1000 items with capacity 100".
# Yes, capacity is fixed at 100.
#
# So we can use C=100.
#
# Heuristic:
# If item > 0.5 * C (i.e., > 50):
#   Use Best Fit (must fit in a tight space, or it won't fit at all).
#   Score: -(bins - item)
# Else:
#   Use Worst Fit (keep space open for larger items).
#   Score: bins - item
#
# This is a common adaptive strategy.
#
# Let's refine the threshold.
# If item is very small, Worst Fit is definitely good.
# If item is large, Best Fit is good.
#
# Let's implement this adaptive strategy.
#
# Safety:
# If no bin has enough capacity, `bins - item` will be negative.
# For Best Fit (minimize waste), we want the smallest positive waste.
# Score = -(bins - item).
# If bins[i] < item, bins[i] - item < 0. Score > 0.
# If bins[i] >= item, bins[i] - item >= 0. Score <= 0.
# We want the highest score.
# If all bins are too small, all scores are positive. The "highest" score corresponds to the *least negative* waste? No.
# If bins[i] = 10, item = 20. Waste = -10. Score = 10.
# If bins[j] = 5, item = 20. Waste = -15. Score = 15.
# We would pick bin j (score 15) over bin i (score 10).
# But neither fits!
# The problem statement says: "scores each bin with enough remaining capacity".
# "each arriving item goes to the highest-scoring bin".
# It implies we should only score bins that fit.
# "Scored by mean percent excess...".
# The interface says: "np.ndarray of one score per bin with enough remaining capacity".
# This phrasing is slightly ambiguous. Does it mean:
# 1. Return an array of the same length, where invalid bins have -inf?
# 2. Return an array of only valid bins?
#
# "one score per bin with enough remaining capacity"
# Usually in these FunSearch/EoH contracts, the array must be the same length as `bins`, and invalid bins should be assigned a score that ensures they are never picked (e.g., -np.inf).
# The seed program `return -(bins - item)` does NOT mask invalid bins.
# If `bins[i] < item`, `bins[i] - item` is negative. `-(bins[i] - item)` is positive.
# If `bins[j] >= item`, `bins[j] - item` is positive. `-(bins[j] - item)` is negative.
# The positive scores (invalid bins) are HIGHER than the negative scores (valid bins).
# So the seed program would pick an INVALID bin if no valid bin exists?
# Or if a valid bin exists, its score is negative. An invalid bin's score is positive.
# So the seed program is actually broken if it doesn't handle the case where no bin fits, OR it assumes a bin always fits?
# "each arriving item goes to the highest-scoring bin".
# If the highest scoring bin is invalid, the item is placed in an invalid bin? That would crash the simulation or count as a failure.
# However, the seed program has a score of 1.0. This implies it works.
# Maybe the evaluator handles the "no fit" case? Or maybe the item sizes are small enough that a bin always fits?
# With C=100 and Weibull, max item size is unbounded theoretically, but practically small.
# However, if a bin has 5 space and item is 10, it doesn't fit.
#
# Let's look at the seed code again.
# `return -(bins - item)`
# If I have bins [5, 20] and item 10.
# Bin 0: 5-10 = -5. Score = 5.
# Bin 1: 20-10 = 10. Score = -10.
# Highest score is 5 (Bin 0).
# Bin 0 does NOT fit item 10.
# If the evaluator places the item in Bin 0, it exceeds capacity.
# This would likely result in an error or a penalty.
# But the score is 1.0 (baseline).
# Perhaps the evaluator masks invalid bins before calling priority?
# "priority(item, bins) ... scores each bin with enough remaining capacity".
# This might imply that `bins` passed to the function ONLY contains bins with enough capacity?
# No, `bins` is `np.ndarray`. If it were filtered, the index would be lost.
#
# Standard interpretation: The function must return scores for ALL bins, but invalid bins must have a score lower than any valid bin.
# The seed program fails this condition if valid bins exist (valid scores are negative, invalid are positive).
# UNLESS: The evaluator adds a large penalty or the logic is different.
# OR: The seed program is actually "Worst Fit" but with a sign flip that makes it "Best Fit" only if we assume the evaluator picks the MINIMUM score?
# "item goes to the highest-scoring bin".
#
# If the seed program is indeed 1.0, maybe the evaluator interprets the scores differently?
# Or maybe I am misinterpreting "highest-scoring".
#
# Let's assume the standard: Highest score wins. Invalid bins must be -inf.
# The seed program `-(bins - item)` does NOT do this.
# If the seed program is 1.0, it must be valid.
# Why?
# Maybe the items are generated such that they always fit in at least one bin?
# And maybe the "invalid" bins (too small) are so rare or their scores are handled?
# If Bin 0 (5) and Bin 1 (20), Item 10.
# Score 0: 5. Score 1: -10.
# Picks Bin 0. Fails.
#
# Is it possible the score is defined such that *lower* is better?
# "Score: mean excess over the L1 lower bound in percent".
# "The evaluator measures the raw value (lower is better)."
# This refers to the FINAL SCORE of the algorithm (excess %).
# The `priority` function returns scores for bins.
# "each arriving item goes to the highest-scoring bin".
#
# If the seed program is `-(bins - item)`, and it works, then either:
# 1. The evaluator picks the LOWEST score? (Contradicts "highest-scoring").
# 2. The evaluator masks invalid bins.
# 3. The seed program is actually not Best Fit in the way I think.
#
# If the evaluator picks the LOWEST score:
# Bin 0 (5): Score 5.
# Bin 1 (20): Score -10.
# Lowest is -10 (Bin 1).
# Bin 1 fits.
# This makes sense!
# If the rule is "lowest score wins", then `-(bins - item)` is Best Fit (minimize waste).
# And invalid bins (positive scores) are never picked if any valid bin (negative score) exists.
# If NO valid bin exists, all scores are positive. The lowest positive score is the "least invalid" bin?
#
# Let's re-read carefully: "each arriving item goes to the highest-scoring bin".
# It explicitly says HIGHEST.
#
# If HIGHEST wins, and seed is `-(bins - item)`:
# Valid bins: Score <= 0.
# Invalid bins: Score > 0.
# Invalid bins win.
# This implies the seed program is BROKEN unless no invalid bins are ever considered (i.e., `bins` is pre-filtered) or the evaluator handles the overflow.
#
# Given the ambiguity, and that I must provide a "stronger algorithm", I should ensure my algorithm is robust.
# If I assume "Highest Score Wins", I MUST mask invalid bins with -inf.
# If I assume "Lowest Score Wins", I MUST mask invalid bins with +inf.
#
# Which one is it?
# "highest-scoring bin" is in the prompt.
# So I will assume Highest Score Wins.
# And I will mask invalid bins with -np.inf.
#
# If the seed program didn't mask, why did it get 1.0?
# Maybe the Weibull items are small enough that `bins` always has a valid bin, and the "invalid" bins are not present in the `bins` array passed to the function?
# i.e., `bins` only contains bins that have been opened and are currently active?
# Even so, an active bin can have remaining capacity < item.
#
# Perhaps the seed program's score of 1.0 is just the baseline, and it's actually performing poorly or crashing, but the "score" is just the ratio to the seed?
# "1.0 equals the seed program".
# If the seed crashes, it scores 0.
# So the seed must not crash.
#
# Hypothesis: The evaluator masks invalid bins before calling `priority`?
# "scores each bin with enough remaining capacity"
# This phrasing in the task description: "Write a Python priority(item, bins) function that scores each bin with enough remaining capacity".
# This could mean: "The function is responsible for scoring only those bins that have enough capacity."
# But it returns an array "of one score per bin".
# If it returns one score per bin, it must cover all bins.
#
# Let's assume the safest path:
# 1. Calculate `waste = bins - item`.
# 2. Identify valid bins: `waste >= 0`.
# 3. Calculate scores for valid bins.
# 4. Set invalid bins to -np.inf.
# 5. Return scores.
#
# This ensures that if a valid bin exists, it is picked (since -inf is lowest).
# If no valid bin exists, all are -inf. The evaluator will likely handle this (open new bin).
#
# Now, what scoring function?
# If Highest Score Wins:
# Best Fit: Minimize waste. Score = -waste. (Higher score for smaller waste).
# Worst Fit: Maximize waste. Score = waste. (Higher score for larger waste).
#
# I will implement **Adaptive Worst/Best Fit**.
#
# Logic:
# C = 100.
# waste = bins - item
# valid = waste >= 0
#
# If item > 50:
#   # Large item, use Best Fit (tightest fit)
#   scores = -waste
# Else:
#   # Small item, use Worst Fit (loosest fit)
#   scores = waste
#
# scores[~valid] = -np.inf
#
# Return scores.

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining capacity after placing item
    waste = bins - item
    
    # Identify valid bins (enough capacity)
    valid = waste >= 0
    
    # Adaptive Heuristic
    # Capacity is 100.
    # If item is large (>50), prefer tightest fit (Best Fit) to maximize utilization.
    # If item is small (<=50), prefer loosest fit (Worst Fit) to preserve space for larger items.
    
    if item > 50.0:
        # Best Fit: Minimize waste.
        # Higher score = Smaller waste.
        # Score = -waste.
        # Since waste >= 0 for valid bins, scores are <= 0.
        # Invalid bins will be set to -inf.
        scores = -waste
    else:
        # Worst Fit: Maximize waste.
        # Higher score = Larger waste.
        # Score = waste.
        # Since waste >= 0 for valid bins, scores are >= 0.
        # Invalid bins will be set to -inf.
        scores = waste
        
    # Mask invalid bins
    scores = np.where(valid, scores, -np.inf)
    
    return scores

# EVOLVE-BLOCK-END
