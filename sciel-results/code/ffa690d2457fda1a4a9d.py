# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global time tracking for deadline guard
_T0 = time.time()
_DEADLINE = 325.0  # 330s limit, 5s margin
_EPS = 1e-12

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Thompson Sampling samples from the posterior distribution of the objective
    function and selects the point that minimizes the sampled value.
    This is a structural change from Expected Improvement (EI), which
    optimizes the expected improvement. TS is known to perform well in
    multimodal landscapes by allowing for more aggressive exploration of
    uncertain regions, potentially escaping local minima.
    
    Args:
        mu: (2000,) posterior mean
        sigma: (2000,) posterior std
        f_best: float, lowest observed value
    
    Returns:
        (2000,) scores. Higher is better.
    """
    # Deadline check
    if time.time() - _T0 > _DEADLINE:
        # Fallback to exploitation-focused score (LCB with beta=0)
        return -(mu - f_best)

    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, _EPS)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, _EPS)
    mu_safe = np.where(np.isfinite(mu), mu, f_best)

    # Generate random samples from the posterior
    # f_sample ~ N(mu, sigma^2)
    # We want to minimize f_sample, so we maximize -f_sample
    # z ~ N(0, 1)
    z = np.random.normal(0, 1, size=mu_safe.shape)
    
    # Sampled objective values
    f_sample = mu_safe - sigma_safe * z
    
    # Score is negative of the sampled value (higher is better)
    scores = -f_sample
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
