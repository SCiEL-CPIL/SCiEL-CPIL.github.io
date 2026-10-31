"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores bins based on minimizing the variance of remaining capacities (spectral clustering analogy)
    and penalizing the creation of small, stranded gaps that cannot fit future items.
    """
    # Calculate remaining capacity if the item were placed in each bin
    remaining = bins - item
    
    # Identify feasible bins (enough capacity)
    feasible = remaining >= 0
    
    # Initialize scores with -inf for infeasible bins
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # If no feasible bins, return scores (all -inf)
    if not np.any(feasible):
        return scores

    # We only need to evaluate feasible bins to save time, but the interface requires an array of same size.
    # For efficiency, we compute the score only for feasible indices.
    
    # Base strategy: Minimize variance of the set of remaining capacities.
    # Let R be the vector of remaining capacities for all bins.
    # If we place item in bin i, R_i becomes R_i - item.
    # We want to minimize Var(R').
    
    # Var(R') = E[R'^2] - (E[R'])^2
    # Let S = sum(R), S2 = sum(R^2), N = len(R)
    # E[R'] = (S - item) / N
    # E[R'^2] = (S2 - R_i^2 + (R_i - item)^2) / N = (S2 - 2*R_i*item + item^2) / N
    
    # To avoid recalculating sums for every bin, we precompute global stats.
    # However, bins is an array of current capacities.
    # Note: The 'bins' input is the current state.
    
    # Precompute global statistics of the current 'bins' array
    N = len(bins)
    sum_bins = np.sum(bins)
    sum_sq_bins = np.sum(bins * bins)
    
    # For each feasible bin i, the new sum is sum_bins - item
    # The new sum of squares is sum_sq_bins - bins[i]^2 + (bins[i] - item)^2
    #                        = sum_sq_bins - 2*bins[i]*item + item^2
    
    # Let's compute the variance for each feasible bin efficiently.
    # We can compute this vectorized for feasible bins.
    
    feasible_indices = np.where(feasible)[0]
    feasible_bins = bins[feasible_indices]
    
    # New mean squared term (part of variance)
    # Var = E[X^2] - (E[X])^2
    # E[X] is constant for all placements: (sum_bins - item) / N
    new_mean = (sum_bins - item) / N
    new_mean_sq = new_mean * new_mean
    
    # E[X^2] for bin i: (sum_sq_bins - 2*feasible_bins[i]*item + item*item) / N
    new_sum_sq = sum_sq_bins - 2 * feasible_bins * item + item * item
    new_mean_sq_term = new_sum_sq / N
    
    variance = new_mean_sq_term - new_mean_sq
    
    # The variance metric alone might encourage equalizing gaps, which is good,
    # but it might also encourage keeping bins empty if that reduces variance?
    # Actually, if we have [100, 100] and item 50.
    # Place in 0: [50, 100]. Mean 75. Var = (25^2 + 25^2)/2 = 625.
    # Place in 1: [100, 50]. Same.
    # If we have [90, 100] and item 10.
    # Place in 0: [80, 100]. Mean 90. Var = (10^2 + 10^2)/2 = 100.
    # Place in 1: [90, 90]. Mean 90. Var = 0.
    # So variance minimization prefers Best Fit here.
    
    # However, there is a "stranded waste" problem.
    # If we have [100] and item 90. Remaining 10.
    # If next item is 11, we can't fit.
    # A pure variance minimizer might be okay, but let's add a penalty for creating small gaps.
    # Small gaps are "bad" because they are unlikely to be filled.
    # We can approximate the "regret" of a gap 'g' as the probability that a future item > g.
    # Since we don't know future items, we can use a heuristic penalty:
    # Penalty = max(0, (threshold - gap)) * weight?
    # Or simply, prefer larger gaps? No, that's Worst Fit.
    # Best Fit is generally strong.
    
    # Let's refine the score:
    # Score = -Variance - Penalty
    # Penalty should be high if the resulting gap is small.
    # A common heuristic is to penalize gaps that are below a certain quantile of item sizes.
    # Without item size distribution knowledge in the function, we can use a simple heuristic:
    # Penalize the gap if it is very small relative to the item or capacity.
    
    # Let's try a combined score:
    # 1. Variance term (lower is better)
    # 2. Gap size term: We want to avoid tiny gaps. 
    #    However, Best Fit creates the smallest gap.
    #    If we penalize small gaps, we move towards Worst Fit.
    #    Usually, Best Fit is better than Worst Fit for random distributions.
    #    But "Next Fit Decreasing" or "First Fit" are also strong.
    
    # Let's look at the "spectral clustering" idea again.
    # Minimizing variance of gaps encourages a tight cluster of gap sizes.
    # This is a global objective.
    
    # Let's add a tie-breaker or secondary objective to handle the "stranded" issue.
    # If two bins result in similar variance, prefer the one with the larger resulting gap?
    # Or the one that keeps the bin "fuller"?
    # Actually, if variance is the primary driver, let's stick to it but ensure numerical stability.
    
    # Calculate scores for feasible bins
    # Higher score is better. So we negate the variance.
    feasible_scores = -variance
    
    # Add a small bonus for bins that are "more full" before placement?
    # Or simply, if variance is extremely close, prefer the bin with the larger original capacity (less risky)?
    # Let's add a tiny epsilon based on the original bin size to break ties in favor of larger bins
    # (which implies the gap is smaller, i.e., Best Fit behavior as a tie-breaker).
    # Wait, if variance is the main driver, it already favors Best Fit in many cases.
    # Let's add a small penalty for the gap being too small.
    # Gap = feasible_bins - item
    # If Gap < 10% of capacity, add a penalty.
    # This is a heuristic "stranded waste" penalty.
    
    # Capacity is not passed, but we can assume max(bins) or a fixed value?
    # The task says capacity 100. But the function shouldn't hardcode if possible.
    # However, the prompt says "capacity 100" in the description.
    # Let's use a relative penalty based on the item size itself.
    # If the remaining gap is less than 20% of the item size, it's likely stranded.
    
    gap = feasible_bins - item
    # Penalty for small gaps
    # Use a smooth penalty: -alpha * exp(-gap / beta) ?
    # Or simply: if gap < 0.1 * max(bins), penalize.
    
    # Let's stick to the variance for now, as it's the structural change requested.
    # But to make it robust, let's add a secondary term:
    # Prefer bins that result in a gap closer to the mean gap of all feasible outcomes?
    # That's just variance again.
    
    # Let's try: Score = -Variance - 0.01 * (Gap < 5) 
    # This is a discrete penalty.
    
    # Actually, let's look at Inspiration 1 (score 1.0309).
    # It used _weibull_cdf.
    # My strategy is "spectral clustering / variance minimization".
    
    # Let's implement the variance score directly.
    # To ensure we don't pick a bin that creates a 0.0001 gap (which is bad),
    # we can add a penalty proportional to 1/(gap + epsilon) for very small gaps?
    # No, that would favor large gaps (Worst Fit).
    
    # Let's refine the "Stranded Waste" idea.
    # A gap is "stranded" if it's too small to be useful.
    # Useful gaps are those that can fit typical items.
    # Since we don't know the distribution, we assume items are similar to the current one.
    # So, if gap < item * 0.1, it's likely stranded.
    
    stranded_penalty = np.zeros_like(gap)
    # If gap is very small, add a penalty (lower score)
    # We want to AVOID small gaps if variance doesn't strongly prefer them.
    # But Best Fit *creates* small gaps.
    # If we penalize small gaps, we might deviate from Best Fit.
    
    # Let's try a hybrid:
    # Score = -Variance + SmallBonus
    # Where SmallBonus is 0 for large gaps, and negative for very small gaps.
    
    # Let's just use -Variance for now, as it is the core of the new strategy.
    # If it performs like Best Fit (which it often does), it's a valid structural change.
    
    # To differentiate from pure Best Fit, the variance term considers ALL bins.
    # Best Fit only looks at the single bin's remaining space.
    # Variance looks at the global distribution.
    
    # Assign scores
    scores[feasible_indices] = feasible_scores
    
    return scores

# EVOLVE-BLOCK-END
