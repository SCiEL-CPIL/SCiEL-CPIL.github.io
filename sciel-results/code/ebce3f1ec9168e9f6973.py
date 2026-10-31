"""Rewritten online bin-packing priority: probabilistic best-fit with polarization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Assumed bin capacity for Weibull benchmark sequences
_CAP = 100.0

# Expected next-item size proxy (tuned to Weibull-like item distributions)
# We aim to leave a gap around this value when feasible
_EXPECTED_NEXT = 15.0

# Sharpness of probabilistic transition for the expected-next-item gap
_SIGMA = 8.0

# Weights balancing immediate fit vs. future flexibility
_W_BESTFIT = 1.0
_W_EXPECTED = 1.2
_W_POL = 0.35
_W_CHEM = 0.5

# Weibull shape and scale parameters for the item distribution
# Typical Weibull for bin packing: k ~ 1.5-2.5, lambda ~ 15-25
_WEIBULL_K = 2.0
_WEIBULL_LAMBDA = 20.0

# Soft-min temperature for smooth polarization
_SOFTMIN_T = 5.0

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _softmin(a, b, t):
    """Smooth approximation of min(a, b) using log-sum-exp."""
    # min(a,b) = -logsumexp(-a, -b) * t / t  (for positive values)
    # Use stable logsumexp
    m = np.maximum(a, b)
    log_sum = m + np.log(np.exp(-np.abs(a - m) / t) + np.exp(-np.abs(b - m) / t)) * t
    # Actually, softmin(a,b) ≈ -t * log(exp(-a/t) + exp(-b/t))
    # = -t * (max(-a/t, -b/t) + log(exp(-a/t - max) + exp(-b/t - max)))
    # Let's use the stable form:
    neg_a = -a / t
    neg_b = -b / t
    m2 = np.maximum(neg_a, neg_b)
    log_sum_exp = m2 + np.log(np.exp(neg_a - m2) + np.exp(neg_b - m2))
    return -t * log_sum_exp


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin that can fit the item.
    The item is placed in the bin with the highest score.

    Scoring design:
      - Base: prefer smaller remaining gaps (best-fit) to reduce fragmentation.
      - Probabilistic term: reward gaps near the expected next item size,
        using a smooth exponential kernel to avoid creating unusable slivers.
      - Polarization term: penalize bins that are near half-full, using a
        smooth soft-min to avoid nonsmoothness.
      - Chemical potential term: Weibull-aware derivative that steers remainders
        toward sizes likely to be filled by future items.
    """
    n = bins.size
    scores = np.full(n, -np.inf, dtype=np.float64)

    if n == 0:
        return scores

    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return scores

    remaining = bins - item
    feasible = remaining >= 0.0

    if not np.any(feasible):
        return scores

    rem = remaining[feasible]

    # 1) Best-fit component: smaller gap is better
    bestfit = -rem * _W_BESTFIT

    # 2) Probabilistic kernel around expected next item
    #    Score peaks when rem == _EXPECTED_NEXT
    z = (rem - _EXPECTED_NEXT) / _SIGMA
    z = np.clip(z, -8.0, 8.0)
    kernel = np.exp(-z * z) * _W_EXPECTED

    # 3) Polarization penalty using soft-min for smoothness
    #    softmin(rem, CAP - rem) is small when near empty or full (good),
    #    large when near half-full (bad). We penalize larger values.
    pol = _softmin(rem, _CAP - rem, _SOFTMIN_T)
    pol_penalty = -pol * _W_POL

    # 4) Chemical potential term: derivative of Weibull log-likelihood w.r.t. remainder
    #    Weibull PDF: f(x) = (k/lambda) * (x/lambda)^(k-1) * exp(-(x/lambda)^k)
    #    log f(x) = log(k/lambda) + (k-1)*log(x/lambda) - (x/lambda)^k
    #    d/dx log f(x) = (k-1)/x - k * x^(k-1) / lambda^k
    #    We want to maximize the probability that a future item fits in the remainder,
    #    so we reward remainders where the Weibull density is high (i.e., likely item sizes).
    #    The "chemical potential" is the negative gradient of the log-density,
    #    so we score by the log-density itself (higher density = better).
    #    Clip rem to avoid log(0)
    rem_clipped = np.clip(rem, 0.1, _CAP - 0.1)
    # log Weibull density at rem_clipped (this represents how likely a future item
    # of this size is, so higher density means the remainder is "useful")
    weibull_logpdf = (np.log(_WEIBULL_K / _WEIBULL_LAMBDA)
                     + (_WEIBULL_K - 1) * np.log(rem_clipped / _WEIBULL_LAMBDA)
                     - (rem_clipped / _WEIBULL_LAMBDA) ** _WEIBULL_K)
    # Normalize and scale
    chem = weibull_logpdf * _W_CHEM

    combined = bestfit + kernel + pol_penalty + chem

    scores[feasible] = combined
    return scores
# EVOLVE-BLOCK-END
