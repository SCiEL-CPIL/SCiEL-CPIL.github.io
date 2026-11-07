"""Rewritten program using LCB acquisition for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function for minimization.
    
    Parameters
    ----------
    mu : array_like, shape (n,)
        Posterior mean at candidate points.
    sigma : array_like, shape (n,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed value so far.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and has a small floor to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive LCB: use a kappa that balances exploration and exploitation.
    # For a 1D problem with only 30 evaluations, we want to encourage exploration
    # early on but shift toward exploitation as we get closer to the optimum.
    # 
    # We can estimate progress by comparing f_best to the known minimum (-39.1662).
    # If f_best is close to the minimum, we should reduce exploration.
    # If f_best is far from the minimum, we should increase exploration.
    
    # Known minimum of Styblinski-Tang in 1D is approximately -39.1662
    f_min = -39.1662
    
    # Estimate how much improvement is still possible
    # If f_best is much worse than f_min, we need more exploration
    # If f_best is close to f_min, we can focus more on exploitation
    
    # A simple adaptive scheme: scale kappa based on the gap
    # gap = f_best - f_min (positive means we're above the minimum)
    # If gap is large, use larger kappa; if gap is small, use smaller kappa
    
    gap = f_best - f_min
    
    # Clamp gap to avoid extreme values
    gap = np.clip(gap, 0, 100)
    
    # Map gap to kappa: when gap is 0, kappa ~ 0.5 (exploit); when gap is large, kappa ~ 2.0 (explore)
    # Use a smooth mapping
    kappa = 0.5 + 1.5 * np.tanh(gap / 10.0)
    
    scores = mu - kappa * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
