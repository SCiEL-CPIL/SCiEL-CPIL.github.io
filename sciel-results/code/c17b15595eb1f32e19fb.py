# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = _T0 + 325.0  # Safety margin vs 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Uses a minimum-variance (maximum uncertainty) strategy to explore
    unexplored regions of the search space. This prevents premature
    convergence and helps locate the global minimum of the sphere function
    by ensuring the GP resolves uncertainty in all regions before exploiting.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Deadline check
    if time.time() > _DEADLINE - 1.0:
        return np.zeros(len(mu))
    
    # Ensure sigma is finite and non-negative
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 0.0)
    
    # The strategy is to maximize the posterior standard deviation.
    # This forces the GP to resolve uncertainty in unexplored regions.
    # For the 1D sphere function, this space-filling behavior helps
    # locate the basin of attraction for the minimum at 0 more reliably
    # than pure exploitation (EI) which might get stuck in local noise
    # or biased regions if initial points are poor.
    
    scores = sigma
    
    # Ensure output is finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
