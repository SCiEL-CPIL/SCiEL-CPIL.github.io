"""Improved program: Best-Fit-Decreasing style heuristic with adaptive scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We track some state to allow a more sophisticated strategy.
# However, since priority() is called per-item and must be stateless or use only inputs,
# we rely on the bins array which reflects current state.

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for placing the given item.
    bins: array of remaining capacities.
    Returns: array of scores, higher is better.
    Only bins with enough capacity are scored; others get -inf.
    """
    remaining = bins
    # Mask bins that can fit the item
    mask = remaining >= item
    
    # If no bin can fit the item, return -inf for all (shouldn't happen in valid pack, but safe)
    if not np.any(mask):
        return -np.inf * np.ones_like(remaining)
    
    # We want to minimize the remaining space after placing the item (Best Fit).
    # But pure Best Fit can be suboptimal. Let's consider a variant:
    # Best Fit Decreasing is known to be good. Here we just have the current item.
    # A common improvement is to prefer bins that are "more full" but not too tight.
    
    # Let's try a weighted combination:
    # 1. Minimize remaining space (Best Fit) -> score = -(remaining - item)
    # 2. Prefer bins with larger initial capacity? We don't have that.
    # 3. Prefer bins where the item fits "nicely" -> maybe avoid fragments?
    
    # The seed program uses: -(bins - item) which is Best Fit.
    # Let's try to tune this.
    
    # One idea: if there are many bins, Best Fit might create many small fragments.
    # Worst Fit (largest remaining) can help spread load but is usually worse for total bins used.
    
    # A hybrid: 
    # If the item is small relative to the bin, maybe use Worst Fit to save the tight bins for larger items?
    # This is a heuristic guess.
    
    # Let's analyze the distribution of remaining capacities.
    # If we place the item in a bin with remaining capacity R, the new remaining is R - item.
    # We want to minimize the total number of bins used.
    # The L1 lower bound is sum(items) / capacity.
    
    # Strategy: 
    # If the item is small, use Worst Fit (maximize remaining space after placement? No, that's just max remaining).
    # Actually, Worst Fit places in the bin with the MOST remaining space.
    # Best Fit places in the bin with the LEAST remaining space (that still fits).
    
    # Research suggests Best Fit Decreasing (BFD) is better than First Fit Decreasing (FFD).
    # Since we don't control the order, we can only choose the bin.
    # If the input order is random, BFD is not directly applicable.
    
    # Let's try a parameterized score:
    # score = -alpha * (remaining - item) + beta * something else?
    
    # Another perspective:
    # We want to keep bins "large" for future large items if possible?
    # This suggests that if the current item is small, we should NOT use the tightest fit, 
    # but rather a bin that is not too small, to preserve the smallest bins for items that fit only there.
    # However, if we use a larger bin for a small item, we fragment a large bin.
    
    # Let's look at the "excess" metric.
    # Excess = (bins_used - L1_bound) / L1_bound.
    # Minimizing bins_used is the goal.
    
    # Simple heuristic: Best Fit is usually very strong.
    # Let's try a slight modification: 
    # If there is a bin that fits the item exactly (remaining == item), use it.
    # Otherwise, use the bin with the smallest remaining capacity (Best Fit).
    # This is exactly the seed.
    
    # What if we prioritize bins that have been "used" less? We don't have that info.
    
    # Let's try a randomized Best Fit? No, deterministic is safer for reproducibility and score stability.
    
    # Let's try an adaptive approach based on the item size relative to the average remaining capacity.
    # If item is small, use Worst Fit?
    # If item is large, use Best Fit?
    
    avg_remaining = np.mean(remaining[mask])
    if avg_remaining > 0:
        ratio = item / avg_remaining
    else:
        ratio = 1.0
    
    # If ratio is small (item is small), maybe use Worst Fit to avoid fragmenting small bins?
    # If ratio is large (item is large), use Best Fit.
    
    # Let's define two scores:
    best_fit_score = -(remaining[mask] - item)
    worst_fit_score = remaining[mask]  # Maximize remaining capacity? No, that's just picking the largest bin.
    
    # Actually, Worst Fit picks the bin with max remaining capacity.
    # Score for Worst Fit: remaining[mask]
    # Score for Best Fit: -(remaining[mask] - item)
    
    # Let's blend them.
    # If item is very small, weight towards Worst Fit.
    # If item is large, weight towards Best Fit.
    
    # We need to map this to a single score array.
    # Let's try: 
    # weight = sigmoid(k * (threshold - ratio))
    # score = weight * worst_fit_score + (1-weight) * best_fit_score
    
    # But we need to be careful with scales.
    # best_fit_score is negative (since remaining >= item).
    # worst_fit_score is positive.
    
    # Let's normalize?
    
    # Alternative: Just stick with Best Fit but break ties differently?
    # The seed program returns -(bins - item).
    # This effectively picks the bin with min(remaining) among those >= item.
    
    # Let's try a different heuristic: 
    # "Worst Fit Decreasing" is not standard.
    # "First Fit Decreasing" is standard.
    
    # What if we look at the second smallest remaining capacity?
    
    # Let's try a simple parameter optimization on a small set of strategies:
    # 1. Best Fit (seed)
    # 2. Worst Fit
    # 3. First Fit (lowest index with enough capacity) - but we don't know indices in a meaningful way for "first".
    
    # Since we don't have history, let's assume the evaluator generates Weibull sequences.
    # Weibull distributions have a shape parameter. If shape < 1, it's decreasing density (more small items).
    # If shape > 1, it's increasing density (more large items).
    
    # For Weibull with capacity 100, items are likely < 100.
    
    # Let's try to improve upon Best Fit by considering the "waste" created.
    # Waste = (remaining - item) for the chosen bin.
    # Total waste = sum of wastes in used bins? No, total waste = sum(capacities of used bins) - sum(items).
    # To minimize bins, we want to maximize the "fill" of each bin used.
    # So, for the current item, we want to place it in a bin that is already highly filled?
    # That means remaining is small. -> Best Fit.
    
    # So Best Fit is theoretically sound for minimizing the number of bins if items are known in advance.
    # Online, it's a good greedy choice.
    
    # Let's check if there's a known improvement for online bin packing.
    # "Next Fit" is bad.
    # "First Fit" is okay.
    # "Best Fit" is generally better than First Fit.
    
    # Maybe the seed program IS the best simple heuristic.
    # How to beat it?
    
    # Perhaps by looking at the distribution of remaining capacities.
    # If many bins have very little remaining capacity, we should be careful not to waste them.
    # But Best Fit already uses the tightest fit, which minimizes waste in that specific bin.
    
    # What if we have a bin with remaining = 50, and item = 10.
    # Best Fit puts it in the bin with remaining=10 (if exists) or 50 if no smaller.
    # If we put it in 50, remaining becomes 40.
    # If we put it in 10, remaining becomes 0.
    # Using the 10-bin is better because it "closes" that bin potentially, or uses up a small fragment.
    
    # So Best Fit is good.
    
    # Let's try a "Harmonic" or "Randomized" variant? No, deterministic required.
    
    # Let's try to optimize the score function slightly.
    # score = -(remaining - item) is linear.
    # What if we use a penalty for large gaps?
    # score = -(remaining - item) - lambda * ((remaining - item)^2)
    # This would prefer even tighter fits if there's a choice between, say, remaining=11 (gap=1) and remaining=12 (gap=2).
    # But Best Fit already picks the smallest remaining. So this only matters if there are ties?
    # No, if remaining=11 and 12 are the two smallest, Best Fit picks 11.
    # The quadratic term doesn't change the order if we are picking the min of remaining.
    # Wait, the score is returned, and the HIGHEST score is chosen.
    # score = -(remaining - item).
    # If remaining=11, score = -(1) = -1.
    # If remaining=12, score = -(2) = -2.
    # -1 > -2, so bin 11 is chosen. Correct.
    
    # So the linear term correctly implements Best Fit.
    
    # How can we deviate from Best Fit to improve?
    # Maybe if the item is very small, using the tightest fit is wasteful of the "bin slot" if the bin could hold more items?
    # But we don't know future items.
    
    # Let's try a "Worst Fit" for small items.
    # Hypothesis: If item is small, it doesn't matter much where it goes. Putting it in a large bin preserves the small bins for items that *must* go there (i.e., large items that only fit in small bins? No, large items need large bins).
    # Actually, large items need large remaining capacity.
    # Small items can fit anywhere.
    # If we put small items in large bins, we fragment large bins.
    # If we put small items in small bins, we fill up small bins.
    # Which is better?
    # If we fill up small bins, we free up the "mind" to focus on large bins for large items.
    # So Best Fit (small bins for small items) seems better.
    
    # Let's look at the Weibull distribution.
    # If the distribution is skewed towards small items, there will be many small items.
    # Best Fit will pack them tightly into the smallest available bins.
    # This leaves the larger bins empty for the few large items.
    # This seems optimal.
    
    # So why is the seed score 1.0? It's the baseline.
    # To improve, we need a better heuristic.
    
    # What if we use "First Fit" with a twist?
    
    # Let's try a different scoring:
    # score = 1 / (remaining - item + epsilon)
    # This strongly favors tight fits.
    # It's equivalent to Best Fit in terms of ordering (monotonic).
    
    # Maybe the issue is tie-breaking?
    # If two bins have the same remaining capacity, which one to pick?
    # It shouldn't matter if capacities are identical.
    
    # Let's consider the possibility that the input `bins` array order matters for some reason?
    # No, the evaluator likely just checks the final number of bins.
    
    # Is it possible that "Best Fit" is not optimal for Weibull sequences?
    # There are papers on "Next Fit" vs "First Fit" vs "Best Fit".
    # Best Fit is generally asymptotically optimal for some distributions?
    
    # Let's try to implement a "Parameterized Best Fit".
    # score = - (remaining - item) / (remaining + epsilon)
    # This normalizes the gap by the bin size.
    # A gap of 1 in a bin of size 10 is 10% waste.
    # A gap of 1 in a bin of size 100 is 1% waste.
    # Best Fit minimizes absolute gap.
    # This new score minimizes relative gap?
    # No, score = -gap/remaining.
    # Higher score means smaller gap/remaining.
    # So it prefers bins where the gap is small relative to the bin size.
    # This is "Best Fit" in a relative sense.
    
    # Let's test this hypothesis mentally.
    # Bin A: rem=11, item=10. gap=1. ratio=1/11 ~ 0.09. score = -0.09.
    # Bin B: rem=101, item=10. gap=91. ratio=91/101 ~ 0.90. score = -0.90.
    # Bin A is chosen. Same as Best Fit.
    
    # What if:
    # Bin A: rem=20, item=10. gap=10. ratio=0.5. score=-0.5.
    # Bin B: rem=11, item=10. gap=1. ratio=0.09. score=-0.09.
    # Bin B is chosen. Same as Best Fit.
    
    # So this doesn't change the preference order compared to Best Fit?
    # Let's check:
    # Best Fit picks min(rem).
    # Relative Best Fit picks min(gap/rem) = min((rem-item)/rem) = min(1 - item/rem) = max(item/rem) = min(rem).
    # Yes, it's identical to Best Fit.
    
    # So we need a different metric.
    
    # What if we prefer to keep bins "open" (i.e., with lots of space) for future large items?
    # This suggests using "Worst Fit" for small items.
    
    # Let's try a blend:
    # If item < threshold, use Worst Fit.
    # Else, use Best Fit.
    
    # Threshold: maybe 10% of capacity? Capacity is 100. So item < 10?
    
    # Let's implement this and hope it improves.
    
    # Smooth transition between Best Fit and a softer strategy for small items.
    # Best Fit: score = -remaining (max score = min remaining)
    # Worst Fit: score = remaining (max score = max remaining)
    # We use a coefficient that transitions from -1 (Best Fit) to 0 (Neutral) for small items.
    # This avoids the aggressive Worst Fit that caused regression in the hard-switch version.
    
    # Parameters tuned to favor Best Fit strongly, with a softening effect for very small items.
    threshold = 2.0
    k = 1.0
    
    # sigmoid(k * (item - threshold)) is ~0 for small items, ~1 for large items.
    # We want coeff to be -1 for large items (Best Fit) and closer to 0 for small items.
    # coeff = -1 + sigmoid(k * (item - threshold))
    # If item is large, sigmoid->1, coeff->0? No.
    # Let's check:
    # item=100, sigmoid(1.0*98) ~ 1. coeff = 0. Score = 0. (Neutral). This is wrong.
    # We want Best Fit for large items.
    # So we want coeff = -1 for large items.
    # sigmoid(k * (item - threshold)) -> 1 for large.
    # So coeff = -sigmoid(...) ?
    # If large, coeff = -1. Score = -remaining. (Best Fit).
    # If small, sigmoid->0, coeff->0. Score = 0. (Neutral).
    
    sig = 1.0 / (1.0 + np.exp(-k * (item - threshold)))
    # For large item, sig ~ 1. coeff = -1.
    # For small item, sig ~ 0. coeff = 0.
    coeff = -sig
    
    scores = coeff * remaining
    
    # Mask out bins that don't fit.
    scores[~mask] = -np.inf
    
    return scores

# EVOLVE-BLOCK-END
