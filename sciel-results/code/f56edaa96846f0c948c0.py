"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We can use a simple, fast heuristic.
# The best-fit is a strong baseline.
# Let's try a variation that considers the "waste" more carefully.
# Or perhaps a combination of best-fit and first-fit-decreasing? No, items arrive online.
#
# A common improvement is "Best Fit Decreasing" but that's offline.
# Online: Best Fit is standard.
# Let's try "Harmonic" or "Next Fit Decreasing" variations?
#
# Actually, the prompt mentions "smooth nonlinear power-law transform".
# The current seed is Best Fit: score = -(bins - item).
# This maximizes the fill.
#
# Let's try a score that penalizes large remaining space more heavily?
# Or one that prefers bins that are "almost full" but not too full?
#
# Let's look at the "inspiration": "smooth nonlinear power-law transform that captures diminishing marginal value of remaining space".
# This suggests that reducing remaining space from 10 to 5 is worth more than 5 to 0?
# Or maybe the value of filling a bin is concave?
#
# If we want to minimize total bins, we want to fill bins as much as possible.
# Best Fit does this greedily.
#
# Let's try a score based on the ratio of item to remaining space?
# Or simply stick to Best Fit but optimize the implementation?
# The task asks for a "stronger algorithm".
#
# What if we use a weighted combination of Best Fit and something else?
#
# Let's consider the L1 lower bound.
# L1 = ceil(sum(items) / capacity).
# We want to minimize (bins_used - L1).
#
# Best Fit is known to be good.
#
# Let's try "Best Fit with a twist":
# Score = -(bins - item) ** alpha
# If alpha > 1, we penalize large remaining space more.
# If alpha < 1, we penalize small remaining space more (prefer to keep space? No, we want to fill).
# Wait, if we want to fill, we want (bins - item) to be small.
# So we want to maximize - (bins - item).
# If we use - (bins - item) ** alpha:
#   alpha=1: Best Fit.
#   alpha=2: Penalizes large remaining space quadratically. This might force items into bins with slightly more space than strictly necessary to avoid leaving a bin with huge space?
#   Actually, Best Fit already picks the smallest remaining space.
#   If there are two bins with remaining space 10 and 20, Best Fit picks 10.
#   With alpha=2, score for 10 is -100, for 20 is -400. Still picks 10.
#   So alpha doesn't change the order if all remaining spaces are positive.
#   It only changes the relative scores, which doesn't matter for argmax.
#
# So we need a different metric.
#
# What about "First Fit"?
# First Fit picks the first bin that fits.
# Best Fit is generally better than First Fit.
#
# What about "Worst Fit"?
# Worst Fit picks the bin with the most remaining space.
# This is usually worse.
#
# Let's try a hybrid:
# If the item is small, use Best Fit?
# If the item is large, use something else?
#
# Actually, a known online algorithm is "Harmonic" or "Next Fit Decreasing" (offline).
# For online, "Best Fit" is strong.
#
# Let's try to improve Best Fit by considering the "efficiency" of the placement.
# Maybe we should avoid creating new bins if possible?
# But we can't control that directly, we just pick a bin.
#
# Let's look at the "power-law" hint again.
# "diminishing marginal value of remaining space"
# This implies that the "cost" of having remaining space is not linear.
# Maybe the cost is convex?
# If cost is convex, we prefer to have one bin with lots of space and one with none, rather than two with moderate space?
# No, we want to minimize bins.
#
# Let's try: Score = -1 / (bins - item + epsilon)
# This strongly prefers bins with very little remaining space.
# This is essentially Best Fit, but with a non-linear preference.
# Since Best Fit picks the minimum (bins - item), and 1/x is monotonic, it picks the same bin.
#
# So any monotonic transformation of the remaining space will yield the same decision as Best Fit.
# To get a different decision, we need a non-monotonic function or a function of other variables.
#
# What other variables do we have?
# Only `item` and `bins` (remaining capacities).
#
# Maybe we should consider the "risk" of not fitting future items?
# If we fill a bin too tightly, we might not fit a future large item.
# If we leave space, we might waste a bin.
#
# A common heuristic is to leave some "buffer" space.
# But we don't know future items.
#
# Let's try "Best Fit Decreasing" style logic but online?
# We can't sort items.
#
# Let's try a simple variation:
# If the remaining space is very close to the item size, prefer it strongly.
# If the remaining space is much larger, prefer it less.
#
# Score = - (bins - item) + lambda * (bins - item)^2 ?
# This is still monotonic for small values.
#
# Let's try: Score = - (bins - item) / (bins)
# This is the fraction of the bin that will be filled.
# Maximizing this is equivalent to minimizing (bins - item) / bins.
# Let f(x) = (x - i) / x = 1 - i/x.
# Maximizing 1 - i/x is equivalent to minimizing i/x, which is equivalent to minimizing x (for fixed i).
# So this is also Best Fit.
#
# It seems hard to beat Best Fit with a simple function of remaining space.
#
# However, Best Fit can be suboptimal.
# Example: Capacity 10.
# Items: 6, 6, 4, 4.
# Best Fit:
# 6 -> Bin 1 (rem 4)
# 6 -> Bin 2 (rem 4)
# 4 -> Bin 1 (rem 0)
# 4 -> Bin 2 (rem 0)
# Total bins: 2. Optimal.
#
# Example: Capacity 10.
# Items: 5, 5, 5, 5.
# Best Fit:
# 5 -> Bin 1 (rem 5)
# 5 -> Bin 1 (rem 0)
# 5 -> Bin 2 (rem 5)
# 5 -> Bin 2 (rem 0)
# Total bins: 2. Optimal.
#
# Example: Capacity 10.
# Items: 3, 3, 3, 3, 3, 3, 3, 3, 3, 3.
# Best Fit:
# 3 -> Bin 1 (7)
# 3 -> Bin 1 (4)
# 3 -> Bin 1 (1)
# 3 -> Bin 2 (7)
# 3 -> Bin 2 (4)
# 3 -> Bin 2 (1)
# 3 -> Bin 3 (7)
# 3 -> Bin 3 (4)
# 3 -> Bin 3 (1)
# 3 -> Bin 4 (7)
# Total bins: 4.
# Optimal: 3 bins (10, 10, 10).
# Best Fit failed here.
#
# Why? Because it filled Bin 1 to 1, then had to open Bin 2.
# If it had spread them out?
# First Fit:
# 3 -> B1 (7)
# 3 -> B1 (4)
# 3 -> B1 (1)
# 3 -> B2 (7)
# 3 -> B2 (4)
# 3 -> B2 (1)
# 3 -> B3 (7)
# 3 -> B3 (4)
# 3 -> B3 (1)
# 3 -> B4 (7)
# Same result.
#
# What if we used "Worst Fit"?
# 3 -> B1 (7)
# 3 -> B1 (4)  (Worst Fit picks max space, so B1 again? No, B1 has 7, others 0. So B1.)
# Wait, Worst Fit picks the bin with the MOST remaining space.
# Initially all bins are empty (infinite space? or we open new bins).
# If we have a pool of bins, Worst Fit would put everything in the first bin until it's full?
# No, if we have existing bins, it picks the one with most space.
# If we start with no bins, we open B1.
# 3 -> B1 (7)
# 3 -> B1 (4) (B1 has 7, others don't exist. If we assume we can open new bins, do we consider empty bins as having capacity 10?
# If we consider empty bins, Worst Fit would always pick an empty bin?
# 3 -> B1 (7)
# 3 -> B2 (7) (B1 has 7, B2 has 10. Pick B2.)
# 3 -> B3 (7)
# ...
# This would open a new bin for every item! Terrible.
#
# So Worst Fit is bad if we consider empty bins.
#
# Back to Best Fit.
#
# Is there a way to avoid the "fragmentation" issue?
# In the 3,3,3... example, Best Fit fragments space.
#
# Maybe we should prefer bins that are "moderately" full?
#
# Let's try a score that is high when the remaining space is close to the item size, but also high when the remaining space is large (to save bins for later)?
# No, that's contradictory.
#
# Let's look at the "L1 lower bound" hint.
# We want to minimize bins used.
#
# What if we use a randomized approach?
# No, deterministic is preferred for reproducibility and speed.
#
# Let's try "Best Fit" but with a small random jitter?
# No.
#
# Let's try a different heuristic: "Next Fit" is bad.
#
# How about "First Fit Decreasing" (FFD) logic?
# We can't sort.
#
# Let's try to improve the 3,3,3 case.
# If we had a rule: "If the remaining space is less than X, don't use it unless it's the only option?"
# No, that would force opening new bins.
#
# Actually, for the 3,3,3 case, the optimal packing is 3-3-4? No, items are 3.
# 3+3+3 = 9. Rem 1.
# 3+3+3 = 9. Rem 1.
# 3+3+3 = 9. Rem 1.
# 3+3+3 = 9. Rem 1.
# 4 bins.
# Wait, 10/3 = 3.33. So 3 items per bin.
# 10 items / 3 items per bin = 3.33 bins. So 4 bins is optimal?
# No, 3*3 = 9. 10-9=1.
# Can we fit 4 items in a bin? 4*3=12 > 10. No.
# So max 3 items per bin.
# 10 items / 3 = 3.33 -> 4 bins.
# So Best Fit was optimal in that case!
#
# Let's find a case where Best Fit is suboptimal.
# Capacity 10.
# Items: 6, 5, 5, 4, 4.
# Best Fit:
# 6 -> B1 (4)
# 5 -> B2 (5)
# 5 -> B2 (0)
# 4 -> B1 (0)
# 4 -> B3 (6)
# Bins: 3.
# Optimal:
# 6+4=10
# 5+5=10
# 4 -> B3
# Bins: 3.
# Same.
#
# Items: 8, 7, 6, 5, 4, 3, 2, 1.
# Sum = 36. L1 = 4.
# Best Fit:
# 8 -> B1 (2)
# 7 -> B2 (3)
# 6 -> B3 (4)
# 5 -> B3 (rem 4? No, 6+5=11 > 10. So B4 (5))
# Wait, B3 has 4 remaining. 5 doesn't fit.
# So 5 -> B4 (5).
# 4 -> B2 (3? No, 7+4=11. B1 (2? No). B3 (4? Yes). B3 becomes 0.
# 3 -> B1 (2? No). B2 (3? Yes). B2 becomes 0.
# 2 -> B1 (2? Yes). B1 becomes 0.
# 1 -> B4 (5? Yes). B4 becomes 4.
# Bins used: 4.
# Optimal:
# 8+2=10
# 7+3=10
# 6+4=10
# 5+1=6
# Bins: 4.
# Same.
#
# It is hard to find a counterexample for Best Fit on small capacities.
#
# However, Best Fit is known to have a competitive ratio of 1.7.
#
# Let's try a slightly different heuristic:
# "Best Fit" is equivalent to minimizing the remaining space.
# What if we minimize the "waste" relative to the bin capacity?
# That's the same.
#
# Let's try "Harmonic" algorithm?
# The Harmonic algorithm partitions bins into classes based on size.
# But we don't know the item sizes in advance.
#
# Let's try a simple improvement:
# If the item is very small, maybe we should prefer bins that are already quite full?
# That's what Best Fit does.
#
# If the item is very large, maybe we should prefer bins that are empty?
# Best Fit does that too (empty bins have max remaining space, so they are last choice).
# Wait, Best Fit picks the MINIMUM remaining space.
# So it prefers bins that are almost full.
# It only opens a new bin if no existing bin fits.
#
# So Best Fit is "fill the holes".
#
# Is there a "fill the bins" strategy?
# That would be Worst Fit (among existing bins, pick the one with most space).
# But as seen, that's bad if we consider empty bins.
# If we restrict to existing bins that fit, Worst Fit might be better?
#
# Let's test Worst Fit (among fitting bins) on the 3,3,3 case.
# Bins: B1(7), B2(7), B3(7)...
# 3 -> B1 (4) (All have 7, pick first? Or random? Let's say first).
# 3 -> B1 (1) (B1 has 4, others 7. Worst Fit picks max space? No, Worst Fit picks max remaining space. So it would pick B2 (7) instead of B1 (4).
# So:
# 3 -> B1 (7) -> 4
# 3 -> B2 (7) -> 4
# 3 -> B3 (7) -> 4
# 3 -> B4 (7) -> 4
# ...
# This opens a new bin for every item until all bins have 4 remaining?
# No, once B1 has 4, B2 has 7, Worst Fit picks B2.
# So it spreads items out evenly.
# This is "Balanced" or "Round Robin" like.
#
# For 3,3,3...:
# B1: 3,3,3 (9, rem 1)
# B2: 3,3,3 (9, rem 1)
# ...
# This is the same as Best Fit in this case?
# Best Fit:
# 3 -> B1 (4)
# 3 -> B1 (1)
# 3 -> B2 (7) -> 4
# 3 -> B2 (4) -> 1
# ...
# Best Fit fills B1 completely before opening B2?
# No, B1 has 1 remaining. 3 doesn't fit. So it opens B2.
# So Best Fit also creates bins with 1 remaining.
#
# So both Best Fit and Worst Fit (among fitting) seem to perform similarly on this case.
#
# Let's try a case where they differ.
# Capacity 10.
# Items: 4, 4, 4, 4, 4.
# Best Fit:
# 4 -> B1 (6)
# 4 -> B1 (2)
# 4 -> B2 (6)
# 4 -> B2 (2)
# 4 -> B3 (6)
# Bins: 3.
# Optimal: 2 bins (8, 8) + 1 bin (4). Total 3.
#
# Worst Fit (among fitting):
# 4 -> B1 (6)
# 4 -> B1 (2) (B1 has 6, others 0. Pick B1? No, if we consider only fitting bins, B1 fits. Are there others? No. So B1.)
# Wait, if we only consider existing bins that fit, and there's only B1, we must pick B1.
# 4 -> B2 (6) (B1 has 2, doesn't fit. Open B2.)
# 4 -> B2 (2)
# 4 -> B3 (6)
# Same result.
#
# It seems Best Fit is very robust.
#
# Let's try to implement a "smooth" version of Best Fit that might handle ties or near-ties better?
# Or just optimize the code for speed?
# The task says "stronger algorithm".
#
# Maybe we can use a look-ahead heuristic?
# No, we only have current state.
#
# Let's try "First Fit" vs "Best Fit".
# First Fit is faster but usually worse.
#
# What if we combine them?
#
# Let's stick with Best Fit but ensure it's implemented efficiently.
#
# The current seed is:
# return -(bins - item)
#
# This returns a score for ALL bins, including those that don't fit.
# The problem statement says: "scores each bin with enough remaining capacity".
# And "the item goes to the highest-scoring bin".
# If a bin doesn't have enough capacity, it should not be chosen.
# So we should set the score of infeasible bins to -infinity.
#
# The seed program does NOT do this!
# `return -(bins - item)`
# If `bins[i] < item`, then `bins[i] - item` is negative, so `-(bins[i] - item)` is positive.
# This means infeasible bins get a POSITIVE score!
# And feasible bins get a NEGATIVE score (since bins[i] >= item, so bins[i]-item >= 0, so score <= 0).
# So the highest score will ALWAYS be an infeasible bin!
#
# This is a BUG in the seed program?
# Or does the evaluator handle this?
# "scores each bin with enough remaining capacity"
# Maybe the evaluator filters out infeasible bins before calling `priority`?
# Or maybe `bins` only contains feasible bins?
#
# If `bins` only contains feasible bins, then `bins - item` is always >= 0.
# Then `-(bins - item)` is <= 0.
# The highest score is the one closest to 0, i.e., smallest remaining space.
# This is Best Fit.
#
# If `bins` contains ALL bins, the seed is broken.
#
# Let's assume the evaluator passes ALL bins, and we must handle feasibility.
#
# So, correct Best Fit:
# scores = np.full_like(bins, -np.inf)
# feasible = bins >= item
# scores[feasible] = -(bins[feasible] - item)
# return scores
#
# This is a clear improvement over the seed if the seed was broken.
# If the seed was not broken (i.e., evaluator filters), this is equivalent.
#
# Let's implement this robust version.
#
# Also, let's add a small epsilon to avoid division by zero or numerical issues? Not needed for subtraction.
#
# Can we do better than Best Fit?
#
# Let's try "Best Fit" with a penalty for opening new bins?
# We can't distinguish "new bin" from "old bin" in the `bins` array, unless we track it.
# But we only get `bins` array.
#
# If we assume that bins with 0 remaining space are "full" and should be ignored?
# They are infeasible anyway.
#
# Let's try a heuristic that prefers bins with remaining space closer to the item size?
# That's Best Fit.
#
# What if we prefer bins that are "empty" (remaining space = capacity) for large items?
# Best Fit does this (empty bins have max remaining space, so they are least preferred among feasible bins? No, they have HIGH remaining space, so LOW score in Best Fit).
# Wait, Best Fit score is `-(remaining)`.
# Empty bin: remaining = 10. Score = -10.
# Half-full bin: remaining = 5. Score = -5.
# So Best Fit prefers the half-full bin.
#
# Is that good?
# If I have an item of size 5.
# Bin A: rem 5. Bin B: rem 10.
# Best Fit picks A.
# This fills A. B remains empty.
# If next item is 5, it goes to B.
# Total 2 bins.
# If I picked B first:
# Item 5 -> B (rem 5).
# Item 5 -> B (rem 0).
# Total 1 bin.
#
# So Best Fit was SUBOPTIMAL here!
#
# Let's verify.
# Capacity 10.
# Items: 5, 5.
# Best Fit:
# 5 -> B1 (rem 5). (Assume B1 is the only bin, or first one).
# Wait, if we start with no bins, we open B1.
# 5 -> B1 (rem 5).
# 5 -> B1 (rem 0).
# Total 1 bin.
#
# My previous example assumed B1 and B2 existed.
# If B1 and B2 are both empty (rem 10).
# 5 -> B1 (rem 5) or B2 (rem 10)?
# Best Fit picks min remaining. Both are 10. Tie. Pick first (B1).
# B1 rem 5.
# 5 -> B1 (rem 0).
# Total 1 bin.
#
# So Best Fit is optimal here.
#
# What if:
# Items: 6, 4, 4, 4.
# B1, B2 empty.
# 6 -> B1 (rem 4).
# 4 -> B1 (rem 0).
# 4 -> B2 (rem 6).
# 4 -> B2 (rem 2).
# Total 2 bins.
# Optimal: 2 bins (10, 8).
#
# Items: 7, 3, 3, 3, 3, 3.
# 7 -> B1 (rem 3).
# 3 -> B1 (rem 0).
# 3 -> B2 (rem 7).
# 3 -> B2 (rem 4).
# 3 -> B2 (rem 1).
# 3 -> B3 (rem 7).
# Total 3 bins.
# Optimal:
# 7+3=10
# 3+3+3=9
# 3 -> B3
# Total 3 bins.
#
# It seems Best Fit is very hard to beat on small examples.
#
# Let's implement the robust Best Fit.
#
# To potentially improve, we can add a small randomization to break ties?
# No, deterministic is better.
#
# We can try "First Fit" if Best Fit is too slow?
# No, Best Fit is O(N) per item.
#
# Let's try a "Weighted Best Fit" where we slightly prefer bins that are not too full?
# No, that would be worse.
#
# Let's just implement the correct Best Fit.
#
# One more idea: "Next Fit" is bad. "First Fit" is okay. "Best Fit" is good.
#
# What if we use "Best Fit" but if the remaining space is very small (e.g., < 10% of capacity), we consider it "wasted" and prefer a less full bin?
# This is a "threshold" heuristic.
#
# Let's try:
# Score = -(bins - item)
# But if (bins - item) < threshold, add a penalty?
# No, we want to maximize score.
#
# Let's stick to the simplest correct implementation.
#
# EVOLVE-BLOCK-END
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining space
    remaining = bins - item
    
    # Initialize scores with -infinity
    scores = np.full_like(remaining, -np.inf, dtype=float)
    
    # Identify feasible bins
    feasible = remaining >= 0
    
    # For feasible bins, use Best Fit (minimize remaining space)
    # Score = -remaining
    scores[feasible] = -remaining[feasible]
    
    return scores
# EVOLVE-BLOCK-END
