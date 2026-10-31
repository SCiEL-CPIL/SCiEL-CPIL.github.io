"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-compute Weibull CDF-related constants.
# The Weibull distribution used for item generation has shape k and scale lambda.
# For a Weibull(k, lambda), the CDF is F(x) = 1 - exp(-(x/lambda)^k).
# The PDF is f(x) = (k/lambda) * (x/lambda)^(k-1) * exp(-(x/lambda)^k).
#
# We want to score bins based on the "expected future fill" or probability that
# a future item fits. A good heuristic is to use the survival function or CDF
# of the remaining space.
#
# Let's assume typical Weibull parameters for bin packing. Common choices are
# k=2 (Rayleigh) or k=1 (Exponential). Let's try to infer or use a reasonable default.
# The task says "five generated Weibull sequences". The parameters are fixed but unknown.
# However, a robust approach is to use a score that works well across a range of Weibull shapes.
#
# The previous successful strategy used a Weibull CDF-based score.
# Let's refine the "effective capacity" idea.
# Score(bin) = - (remaining_capacity - E[next_item | next_item <= remaining_capacity])
# This is complex. A simpler proxy:
# Score(bin) = - remaining_capacity * (1 - F(remaining_capacity))
# Where F is the CDF of the item size distribution.
# If F(r) is high (small r), then 1-F(r) is low, so the penalty is small? No.
# We want to fill bins. We want to choose bins where the remaining space is "useful".
# If remaining space r is very small, it's likely to be wasted (if next item > r).
# If remaining space r is large, it's likely to be filled.
#
# Let's use the probability that a random item fits in the remaining space: P(X <= r) = F(r).
# We want to maximize this probability? Or minimize the waste?
# If we put an item in a bin with remaining space r, the new remaining space is r - item.
#
# Let's stick to the measured improvement: Weibull CDF-based score.
# The score should be higher for bins that are "better".
# In the seed, higher score is better.
# Seed: return -(bins - item). This favors bins with smallest remaining space (Best Fit).
#
# Let's try: score = F(remaining_space)
# If remaining space is large, F(r) is close to 1. Score is high.
# If remaining space is small, F(r) is close to 0. Score is low.
# This would favor Largest Fit, which is usually worse than Best Fit.
#
# Let's try: score = - (1 - F(remaining_space))
# If remaining space is large, 1-F(r) is small. Score is close to 0 (high).
# If remaining space is small, 1-F(r) is large. Score is very negative (low).
# This still favors large remaining space.
#
# Maybe we should penalize waste.
# Expected waste if we don't fill the bin?
#
# Let's look at the "effective capacity" strategy from the directive:
# "scoring each bin based on the negative of the 'effective capacity' defined as the remaining space minus the expected size of the next item drawn from the Weibull distribution"
# Effective Capacity = r - E[X]
# Score = - (r - E[X]) = E[X] - r
# This is equivalent to Best Fit if E[X] is constant!
# Because maximizing E[X] - r is the same as minimizing r.
# So that strategy is just Best Fit with a constant offset.
#
# Wait, the directive says "thereby prioritizing bins that maximize the probability of accepting a second consecutive item."
# This implies a more complex calculation.
#
# Let's try a score based on the density of the Weibull distribution at the remaining space.
# f(r) = (k/lambda) * (r/lambda)^(k-1) * exp(-(r/lambda)^k)
# If we place an item in a bin with remaining space r, the "value" of that space is how likely it is to be used by future items.
#
# Let's try a hybrid:
# score = -r + alpha * f(r)
# Or score = -r / (1 + f(r))
#
# Let's go back to the measured finding: "Weibull CDF-based score that penalizes large remainders more heavily".
# Penalty for remainder r: P(r) = 1 - F(r) (Probability that a random item does NOT fit)
# Score = - P(r) = F(r)
# As analyzed, F(r) is increasing, so this favors large r.
#
# What if we score based on the "waste" relative to the distribution?
#
# Let's try: score = - (r * (1 - F(r)))
# This is the expected excess of r over a random item X, given that X < r? No.
# E[max(0, r-X)] = integral_0^r (r-x) f(x) dx + integral_r^inf (r-x) f(x) dx (which is 0)
# = r * F(r) - integral_0^r x f(x) dx
#
# Let's simply use the Best Fit heuristic but with a slight modification based on the Weibull PDF.
# If the PDF is high at r, it means items of size r are common. So having a remainder of r is "good" because it's likely to be filled by a future item of size r?
# No, if we have remainder r, we want an item of size <= r.
#
# Let's try: score = -r + beta * f(r)
# With small beta.
#
# Actually, the simplest improvement over Best Fit is often to use a "Worst Fit Decreasing" variant or "First Fit".
# But Best Fit is usually strong.
#
# Let's try a score that considers the "tightness" of the fit.
# tightness = item / r
# score = tightness
# This favors bins where the item fills a large portion of the remaining space.
# This is equivalent to minimizing r / item, or minimizing r.
# It's the same as Best Fit.
#
# Let's try: score = - (r - item)^2
# This is quadratic Best Fit.
#
# Let's try the Weibull CDF penalty again, but inverted.
# We want to avoid bins where the remaining space is "unfillable".
# Unfillable probability = 1 - F(r).
# Score = - (1 - F(r))
# This favors large r.
#
# What if we combine Best Fit and Large Fit?
# score = -r + gamma * r
# = (gamma - 1) r
# If gamma < 1, it's Best Fit.
#
# Let's look at the "effective capacity" again.
# E[X | X <= r] = r - E[max(0, r-X)] / F(r)
#
# Let's try a score based on the mean of the truncated distribution.
# If we assume the next item is drawn from the Weibull distribution, and we want to maximize the expected fill of the bin over the next few items, it's a complex MDP.
#
# Let's try a simple heuristic:
# score = -r + 0.1 * f(r)
# where f(r) is the Weibull PDF.
#
# We need to assume parameters for Weibull.
# Let's assume k=2, lambda=10 (mean = lambda * Gamma(1+1/k) = 10 * Gamma(1.5) = 10 * 0.886 = 8.86).
# If items are around 9, and capacity is 100, we can fit ~11 items per bin.
#
# Let's implement a score that uses the Weibull CDF to penalize remainders that are likely to be left empty.
# Penalty(r) = 1 - F(r)
# Score = - Penalty(r) = F(r)
# This favors large remainders.
#
# Let's try: Score = F(r) / r
# For small r, F(r) is small, so F(r)/r is finite (limit is f(0)).
# For large r, F(r) is 1, so F(r)/r is small.
# This favors medium r.
#
# Let's try: Score = - (r - F(r))
#
# Given the uncertainty, let's stick to a robust Best Fit variant.
# The "measured" improvement was 1.031 with a Weibull CDF-based score.
# Let's try to replicate that more closely.
# "penalizes large remainders more heavily"
# Penalty(r) = r^2 ?
# Score = -r^2
#
# Let's try:
# k = 2.0
# lam = 10.0
# F = lambda r: 1 - np.exp(-(r/lam)**k)
# score = F(remaining)
#
# Wait, if score = F(remaining), and F is increasing, we pick the bin with the LARGEST remaining space.
# That is Worst Fit.
# Worst Fit is generally worse than Best Fit for bin packing.
#
# Maybe the score was - (1 - F(remaining))?
# That is F(remaining). Same thing.
#
# Maybe the score was based on the PDF?
# score = f(remaining)
# For k=2, lam=10, f(r) = (r/100) * exp(-(r/10)^2)
# This peaks at r=10.
# So it favors remainders around 10.
# This makes sense! It favors "medium" remainders that are likely to be filled by a typical item.
#
# Let's try score = f(remaining)
#
# We need to handle r=0. f(0) = 0 for k>1.
#
# Let's implement this.
#
# Parameters: k=2, lam=10.
#
# def priority(item, bins):
#     remaining = bins - item
#     valid = remaining >= 0
#     scores = np.zeros_like(remaining)
#     r = remaining[valid]
#     # Weibull PDF
#     k = 2.0
#     lam = 10.0
#     f = (k/lam) * (r/lam)**(k-1) * np.exp(-(r/lam)**k)
#     scores[valid] = f
#     # For invalid bins, set score to -inf or very low
#     scores[~valid] = -np.inf
#     return scores
#
# This favors bins with remaining space around the mode of the Weibull distribution.
# This is a "Mode Fit" heuristic.
#
# Let's refine the parameters.
# If the mean item size is ~9, mode is ~ sqrt(100 * (2-1)/2) ? No.
# Mode of Weibull(k, lam) is lam * ((k-1)/k)^(1/k) for k>1.
# For k=2, mode = lam * (1/2)^(1/2) = lam * 0.707.
# If lam=10, mode=7.07.
#
# Let's try k=2, lam=10.
#
# Also, add a small Best Fit component to break ties or handle edge cases.
# score = f(r) - 0.01 * r
#
# Let's try this.
# EVOLVE-BLOCK-END
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid = remaining >= 0
    scores = np.full_like(remaining, -np.inf, dtype=float)
    
    if np.any(valid):
        r = remaining[valid]
        
        # Weibull PDF parameters
        k = 2.0
        lam = 10.0
        
        # Calculate PDF: f(x) = (k/lam) * (x/lam)^(k-1) * exp(-(x/lam)^k)
        # Handle r=0 case to avoid issues (though f(0)=0 for k>1)
        with np.errstate(divide='ignore', invalid='ignore'):
            pdf = (k / lam) * (r / lam) ** (k - 1) * np.exp(-(r / lam) ** k)
        
        # Ensure pdf is non-negative and finite
        pdf = np.maximum(pdf, 0.0)
        
        # Combine with a small Best Fit penalty to prefer smaller remainders among similar PDFs
        # The PDF peaks at the mode, so this favors remainders near the mode.
        # Adding a small negative term for r ensures that if two bins have similar PDFs,
        # the one with slightly smaller r is preferred (closer to Best Fit).
        scores[valid] = pdf - 0.01 * r
    
    return scores
# EVOLVE-BLOCK-END
