"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses the 0.1st percentile of the posterior predictive distribution (Q0.1).
    This targets the lower tail of the distribution, encouraging exploration
    in regions with high uncertainty where the worst-case outcome is still
    potentially better than the current best, while also exploiting regions
    with low mean.
    """
    # Safety: ensure finite outputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle any non-finite values in sigma by clamping
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-12)
    
    # Calculate the 0.1st percentile of the normal distribution N(mu, sigma^2)
    # For a normal distribution, the q-th percentile is mu + z_q * sigma
    # where z_q is the q-th quantile of the standard normal distribution.
    # We want the 0.1st percentile (q=0.001).
    # z_0.001 is approximately -3.09.
    # Using scipy.stats.norm.ppf would be more precise, but to avoid import overhead
    # and potential issues, we can use a hardcoded value or calculate it.
    # However, the prompt says SciPy is available. Let's use it for precision if possible,
    # or just use the approximation. The approximation -3.09 is standard.
    # Let's use a slightly more conservative quantile to ensure we don't get stuck in
    # high noise regions that are actually bad. 0.1% is quite extreme.
    # A common choice for quantile acquisition is the median (0.5) or a lower quantile.
    # The directive specifies 0.1st percentile.
    
    # z-score for 0.1% (0.001) quantile of standard normal
    # From standard normal table: P(Z < -3.09) ≈ 0.001
    z_q = -3.09
    
    # Quantile value
    score = mu + z_q * sigma
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
