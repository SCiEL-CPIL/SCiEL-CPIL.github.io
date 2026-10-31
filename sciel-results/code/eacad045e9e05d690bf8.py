"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-compute a lookup table for the Weibull CDF-based regret score.
# The items are drawn from a Weibull distribution. For online bin packing with
# capacity 100, we need to model the "regret" of leaving a remainder r in a bin.
# The regret is high if r is unlikely to be filled by future items.
# Future items are Weibull distributed. Let's assume shape k=1.5, scale lambda=15
# (mean ~15, so items are typically 10-20, max ~50-60).
# Actually, the task says "five generated Weibull sequences". We don't know the exact parameters.
# However, standard Weibull for bin packing often uses k=1 (Exponential) or similar.
# Let's assume a Weibull distribution that fits typical bin packing benchmarks.
# If mean item size is ~10-20, and capacity is 100, we can estimate parameters.
# Let's try to be adaptive or use a robust heuristic.
#
# The strategy suggested is: score = -CDF(remainder) for the distribution of item sums?
# No, the remainder is a single value. We want to know if a future item of size x can fit.
# If we place item 'item' in a bin with remaining 'r', the new remainder is 'r - item'.
# We want 'r - item' to be "good".
# A remainder 'rem' is good if it's likely to be filled by a future item.
# This is tricky because future items are random.
# A common heuristic is Best-Fit Decreasing (offline) or Best-Fit (online).
# Best-Fit minimizes the remaining space, which is equivalent to maximizing the fit.
# The current seed is Best-Fit: return -(bins - item). This prefers the bin where (bins - item) is smallest.
#
# Let's look at the "regret" idea.
# If we leave a remainder 'rem', we waste 'rem' capacity if no future item fits.
# The probability that a future item fits in 'rem' is P(Item <= rem).
# If we assume items are Weibull(k, lambda), P(Item <= rem) = 1 - exp(-(rem/lambda)^k).
# We want to maximize the chance of filling the remainder? Or minimize the expected waste?
# Expected waste = rem * P(Item > rem) + ... (complex).
# Simpler: We want 'rem' to be small (Best Fit) OR 'rem' to be a value that is likely to be exactly matched?
#
# The prompt suggests: "score as the negative of the cumulative distribution function".
# Which CDF?
# If we score by -CDF(rem), we prefer small rem (since CDF is increasing). This is similar to Best-Fit.
# But maybe we should consider the distribution of *items* to see if 'rem' is a "good" target.
#
# Let's try a hybrid:
# 1. Best-Fit is a strong baseline.
# 2. First-Fit Decreasing is good offline.
# 3. Next-Fit is bad.
# 4. "Harmonic" or "Random" are sometimes competitive.
#
# The directive says: "Model the problem as a Markov Decision Process... score each bin based on the expected future regret... approximated by the probability that this remainder cannot be perfectly filled".
#
# Let's assume the item distribution is Weibull with shape k=1.0 (Exponential) or k=2.0.
# Given capacity 100, if mean item size is 10, lambda=10.
# If mean item size is 20, lambda=20.
# Without knowing the exact distribution, we can try to estimate it from the first few items?
# No, the function is stateless per call (except for bins). We can't easily maintain state across calls in this interface unless we use global variables.
#
# Let's use a global variable to track item statistics.
# We can estimate the mean and variance of the items seen so far.
# Then, we can approximate the distribution of future items.
#
# Strategy:
# 1. Maintain a running mean and count of items.
# 2. Estimate a Weibull scale parameter lambda from the mean.
# 3. For each bin with enough capacity, calculate the new remainder rem = bins[i] - item.
# 4. Calculate a "regret" score.
#    - If rem is very small, it's good (hard to waste).
#    - If rem is large, it's bad unless it's likely to be filled.
#    - Probability that a future item fits in rem: P(X <= rem).
#    - If we assume Exponential (Weibull k=1), P(X <= rem) = 1 - exp(-rem/lambda).
#    - We want to maximize P(X <= rem)? Or minimize the expected leftover?
#    - Expected leftover if we stop here: rem.
#    - But we might fill it.
#    - Let's use a score that combines Best-Fit (small rem) with "Fillability" (high P(X <= rem)).
#    - Actually, if rem is small, P(X <= rem) is small (for positive items). So these conflict.
#    - Best-Fit works because small rem means less waste *if* it can't be filled.
#    - If rem is large, it *can* be filled, but it's risky.
#    -
#    - Let's try: Score = -rem + alpha * P(X <= rem) * rem?
#    - Or simply: Score = -rem / (1 + P(X > rem))?
#    -
#    - Let's stick to the prompt's hint: "score as the negative of the cumulative distribution function".
#    - Which CDF? The CDF of the item distribution evaluated at the remainder?
#    - score = -CDF(rem).
#    - If rem is small, CDF(rem) is small, so -CDF(rem) is close to 0 (high score? No, we want highest score).
#    - Wait, the item goes to the *highest* scoring bin.
#    - If score = -CDF(rem), and CDF is increasing, then smaller rem -> smaller CDF -> less negative score -> higher score.
#    - So -CDF(rem) prefers smaller remainders. This is essentially Best-Fit, but with a non-linear scaling.
#    - For small rem, CDF(rem) ~ rem/lambda (for Exp). So -CDF(rem) ~ -rem/lambda. Linear.
#    - For large rem, CDF(rem) -> 1. So -CDF(rem) -> -1.
#    - This compresses the scores for large remainders.
#    -
#    - Let's implement this with an estimated Weibull/Exponential distribution.
#
# We need to handle the case where no bins have enough capacity.
# The function returns an array of scores. The evaluator picks the max.
# If no bin has enough capacity, we should return a very low score for all, or handle it.
# The problem states "each arriving item goes to the highest-scoring bin".
# Implicitly, we assume there is always a bin with enough capacity?
# "scores each bin with enough remaining capacity".
# If no bin has enough capacity, the item cannot be placed.
# The interface says "return np.ndarray of one score per bin with enough remaining capacity".
# Wait, "one score per bin with enough remaining capacity".
# Does the array length change?
# "priority(item: float, bins: np.ndarray) -> np.ndarray of one score per bin"
# The signature says "one score per bin". It doesn't say "per bin with enough capacity".
# The text says: "scores each bin with enough remaining capacity".
# Usually, this means we return scores for ALL bins, but bins with insufficient capacity get a very low score (e.g., -inf) so they are never chosen.
# Let's assume we return an array of the same length as `bins`.
#
# Implementation:
# 1. Update global statistics with `item`.
# 2. Estimate lambda.
# 3. For each bin, if bins[i] >= item, calculate rem = bins[i] - item.
#    Calculate score = -CDF(rem).
#    If bins[i] < item, score = -inf.
# 4. Return scores.

_item_count = 0
_item_sum = 0.0
_item_sum_sq = 0.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _item_count, _item_sum, _item_sum_sq
    
    # Update statistics
    _item_count += 1
    _item_sum += item
    _item_sum_sq += item * item
    
    # Estimate parameters
    if _item_count < 2:
        # Not enough data, default to Best-Fit
        scores = np.full(len(bins), -np.inf)
        valid = bins >= item
        if np.any(valid):
            # Best Fit: minimize remaining
            scores[valid] = -(bins[valid] - item)
        return scores
    
    mean = _item_sum / _item_count
    var = (_item_sum_sq / _item_count) - (mean * mean)
    if var < 0:
        var = 0.0
    std = np.sqrt(var)
    
    # Assume Weibull distribution.
    # For Weibull, Mean = lambda * Gamma(1 + 1/k)
    # Var = lambda^2 * (Gamma(1 + 2/k) - Gamma(1 + 1/k)^2)
    # Coefficient of Variation (CV) = std/mean.
    # For k=1 (Exp), CV = 1.
    # For k=2, CV ~ 0.6.
    # For k=4, CV ~ 0.4.
    # We can estimate k from CV.
    # Approximation: k ≈ 3.6 / (CV^2 + 0.1) is a rough heuristic?
    # Or just assume k=1.5 or k=1.0 for simplicity.
    # Let's assume k=1.0 (Exponential) as it's a common worst-case/average-case for random arrivals.
    # If CV is significantly different, we might adjust.
    # Let's stick to Exponential (k=1) for robustness, lambda = mean.
    
    k = 1.0
    lambda_ = mean / (np.exp(np.log(1 + 1/k) * 0 + 0)) # Gamma(1+1/k) for k=1 is Gamma(2)=1. So lambda=mean.
    # Actually, for k=1, Gamma(2)=1. So lambda = mean.
    
    # CDF of Weibull: 1 - exp(-(x/lambda)^k)
    # We want score = -CDF(rem).
    # CDF(rem) = 1 - exp(-(rem/lambda)^k)
    # score = -(1 - exp(-(rem/lambda)^k)) = exp(-(rem/lambda)^k) - 1
    
    scores = np.full(len(bins), -np.inf)
    valid = bins >= item
    if np.any(valid):
        rems = bins[valid] - item
        # Avoid division by zero if mean is 0 (shouldn't happen)
        if lambda_ > 0:
            # Use log-sum-exp or just direct calculation
            # exp(-(rem/lambda)^k) - 1
            # For small rem, this is approx -rem/lambda.
            # For large rem, this is approx -1.
            # This prefers small rems.
            scores[valid] = np.exp(-np.power(rems / lambda_, k)) - 1.0
        else:
            # Fallback to Best Fit
            scores[valid] = -rems
            
    return scores
# EVOLVE-BLOCK-END
