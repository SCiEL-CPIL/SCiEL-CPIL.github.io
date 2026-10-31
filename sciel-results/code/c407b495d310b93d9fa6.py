"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import weibull_min
import time

# We use a fluid-flow-inspired heuristic with a Weibull CDF-based penalty.
# The idea is to score bins based on how "efficiently" they use the remaining capacity
# relative to the item size, while accounting for the statistical properties of future items.
#
# Key insight: For Weibull-distributed items, the probability that a future item can fit
# in a remaining space R is F(R) where F is the CDF. We want to minimize the expected
# wasted space, which is E[max(0, R - Item)] = integral_0^R F(t) dt.
#
# For a Weibull distribution with shape k and scale lambda, F(t) = 1 - exp(-(t/lambda)^k).
#
# The score should favor bins where the remaining space R has low expected waste.
# Expected waste = R - E[min(R, Item)] = R - integral_0^R (1 - F(t)) dt
#              = integral_0^R F(t) dt
#
# For Weibull: integral_0^R (1 - exp(-(t/lambda)^k)) dt
#             = R - lambda * integral_0^{(R/lambda)^k} u^{1/k - 1} e^{-u} du  (with substitution u = (t/lambda)^k)
#             = R - lambda * gamma(1 + 1/k, (R/lambda)^k)
# where gamma(a, x) is the upper incomplete gamma function... wait, let me recalculate.
#
# Actually: E[min(R, Item)] = integral_0^R P(Item > t) dt = integral_0^R exp(-(t/lambda)^k) dt
# Let u = (t/lambda)^k, then t = lambda * u^{1/k}, dt = lambda/k * u^{1/k - 1} du
# integral_0^R exp(-(t/lambda)^k) dt = lambda/k * integral_0^{(R/lambda)^k} u^{1/k - 1} e^{-u} du
#                                   = lambda/k * gamma(1/k, 0, (R/lambda)^k)
#                                   = lambda/k * lower_gamma(1/k, (R/lambda)^k)
# where lower_gamma(a, x) = integral_0^x t^{a-1} e^{-t} dt
#
# So E[Waste] = R - lambda/k * lower_gamma(1/k, (R/lambda)^k)
#
# We want to minimize E[Waste], so score = -E[Waste].
#
# For Weibull sequences with capacity 100, we need to estimate k and lambda.
# Typical Weibull parameters for bin packing: k around 1-2, lambda around 30-50.
# Let's use k=1.5, lambda=40 as reasonable defaults, or better, we can make this adaptive.
#
# Actually, for simplicity and robustness, let's use a smooth approximation.
# The key property: for small R, waste is small; for large R, waste grows roughly linearly
# but with a shape determined by the Weibull parameters.
#
# A simpler approach: use the Weibull CDF to compute a penalty that is:
# - Low for R values that are likely to be filled (high F(R) means many items fit, so waste is low)
# - High for R values that are unlikely to be filled
#
# penalty(R) = R * (1 - F(R))^alpha for some alpha, or simply the expected waste formula.
#
# Let's use the expected waste formula with scipy.
# For performance, we can precompute or use a vectorized approach.
#
# Parameters: We'll use k=1.5 (shape) and lambda=40 (scale) as reasonable defaults for
# Weibull items with capacity 100. The actual sequences may vary, but this should be
# in the right ballpark.
#
# For large items (>50), we previously used Worst-Fit. Let's keep a hybrid:
# - For small items: use Weibull-based expected waste minimization
# - For large items: use Worst-Fit (maximize remaining space)
#
# Actually, let's test if the Weibull-based score alone is sufficient for all item sizes.
# For large R (large remaining space), the expected waste will be high, so we'll prefer
# bins with smaller R, which is Best-Fit-like. But for very large items, we might want
# to keep the space open.
#
# Let's use a smooth blend: score = -expected_waste(R) for all items, but with a
# correction for large items that favors keeping space open.
#
# Actually, the previous hybrid worked well. Let's refine it:
# For small items (item <= 50): score = -E[Waste(R)] using Weibull
# For large items (item > 50): score = R (Worst-Fit)
#
# Let's implement this.

def _weibull_expected_waste(R, k=1.5, lam=40.0):
    """
    Compute expected waste for remaining space R given Weibull(k, lam) future items.
    E[Waste] = integral_0^R F(t) dt = R - E[min(R, Item)]
    E[min(R, Item)] = integral_0^R P(Item > t) dt = integral_0^R exp(-(t/lam)^k) dt
    """
    R = np.asarray(R, dtype=float)
    R = np.maximum(R, 0.0)
    
    # For R = 0, waste is 0
    # For R > 0:
    # E[min(R, Item)] = lam/k * lower_gamma(1/k, (R/lam)^k)
    # E[Waste] = R - lam/k * lower_gamma(1/k, (R/lam)^k)
    
    # Use scipy.special.lowergamma or gammainc
    from scipy.special import gammainc
    
    # lower_gamma(a, x) = gamma(a) * gammainc(a, x)
    # But we need the regularized lower incomplete gamma function.
    # scipy.special.gammainc(a, x) = lower_gamma(a, x) / gamma(a)
    # So lower_gamma(a, x) = gamma(a) * gammainc(a, x)
    
    from scipy.special import gamma as gamma_func
    
    a = 1.0 / k
    x = (R / lam) ** k
    
    # Handle R=0 case
    # For R=0, x=0, gammainc(a, 0) = 0, so E[min] = 0, waste = 0. Good.
    
    # Compute E[min(R, Item)]
    emmin = (lam / k) * gamma_func(a) * gammainc(a, x)
    
    # E[Waste] = R - E[min]
    waste = R - emmin
    
    # Clip to avoid negative values due to numerical issues
    waste = np.maximum(waste, 0.0)
    
    return waste

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    n_bins = len(bins)
    scores = np.full(n_bins, -1e9)
    
    valid = bins >= item
    
    if not np.any(valid):
        return scores
    
    R = bins[valid] - item
    
    # Threshold for switching between Best-Fit and Worst-Fit
    # Capacity is 100.
    threshold = 50.0
    
    if item > threshold:
        # Large item: Use Worst-Fit (maximize remaining space)
        scores[valid] = R
    else:
        # Small item: Use Weibull-based expected waste minimization
        # Score = -E[Waste(R)]
        waste = _weibull_expected_waste(R, k=1.5, lam=40.0)
        scores[valid] = -waste
        
    return scores
# EVOLVE-BLOCK-END
