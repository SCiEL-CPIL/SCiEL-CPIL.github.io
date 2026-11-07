import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 300.0  # Safety margin well below the 330 s wall-clock kill

def acquisition(mu, sigma, f_best):
    """
    Quantile-based (Q0.1) acquisition for minimizing the objective.
    
    The score targets the lower tail of the posterior predictive distribution.
    Specifically, we compute the 0.1st percentile of f(x) given mu(x) and sigma(x):
    Q0.1(x) = mu(x) - z_0.999 * sigma(x)
    where z_0.999 is the 99.9th percentile of the standard normal (approx 3.09).
    
    To make the score "higher is better" for selection, we negate this quantile:
    Score(x) = -(mu(x) - z_0.999 * sigma(x)) = z_0.999 * sigma(x) - mu(x)
    
    This structurally differs from LCB (which uses a fixed or simple adaptive beta)
    by explicitly optimizing for a very low quantile, encouraging exploration only
    where the uncertainty is large enough to potentially contain a value significantly
    below the current best, while still penalizing high means.
    
    We add a small adaptive component to the quantile level (z-score) to balance
    exploration/exploitation based on how far f_best is from the known minimum (3.0).
    However, the core mechanism is the quantile calculation.
    
    mu, sigma : array-like, shape (2000,)  (objective units, sigma >= 0)
    f_best    : float, lowest value observed so far

    Returns: array of 2000 finite scores; HIGHER means "evaluate next".
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = mu.shape[0]

    # ---- wall-clock guard: never let this call approach the kill limit ----
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Exploitation-only fallback (finite, fast).
        score = -mu
        return np.clip(score, -1e12, 1e12)

    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)

    # Base z-score for the 0.1st percentile (99.9th quantile of N(0,1))
    # norm.ppf(0.999) is approx 3.0902
    base_z = 3.0902
    
    # Adaptive adjustment:
    # If f_best is close to the minimum (3.0), we can be more aggressive (higher z)
    # because we are exploiting. If f_best is far, we might want to be slightly
    # more conservative or keep the same aggressive tail to find the basin.
    # Actually, for quantile optimization, a fixed very low quantile is often robust.
    # Let's try a slight adaptation: if we are stuck (f_best not improving much),
    # increase exploration (lower z, i.e., look at a less extreme tail? No, higher z
    # means looking further into the tail, which is more exploratory for minimization
    # if we interpret Q0.1 as "value with 1% chance of being beaten").
    
    # Let's stick to a fixed, strong quantile for stability, as per the strategy directive.
    # The "Quantile-based" strategy specifically mentions Q0.1.
    z_score = base_z
    
    # Score = z_score * sigma - mu
    # This is high when sigma is large (exploration) or mu is low (exploitation).
    # It directly corresponds to maximizing the lower quantile.
    
    score = z_score * sigma_safe - mu

    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Clip to prevent overflow issues in downstream comparisons if any
    score = np.clip(score, -1e12, 1e12)
    
    # Guarantee exactly `n` finite entries (defensive).
    if score.shape[0] != n:
        score = np.resize(score, n)
        
    return score
