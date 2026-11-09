"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling with adaptive exploration scaling for Bayesian optimization.
    
    Combines Thompson Sampling (stochastic posterior sampling) with an adaptive
    exploration bonus that scales with local uncertainty relative to the global
    uncertainty. This provides natural exploration-exploitation balance while
    amplifying exploration in high-uncertainty regions.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        Mean of the GP posterior at candidate points (in objective units).
    sigma : ndarray, shape (n,)
        Standard deviation of the GP posterior at candidate points (in objective units).
    f_best : float
        The lowest observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores; higher means evaluate next.
    """
    n = len(mu)
    
    # Safety: ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-12)
    
    # Thompson Sampling: draw a single scalar standard normal sample
    # Using a single scalar z for all points creates correlation in the
    # sampled function, which can be beneficial for smooth landscapes.
    # But for better exploration diversity, use per-point samples.
    z = np.random.standard_normal(n)
    
    # Posterior predictive samples: y = mu + sigma * z
    # For minimization, we want to find points where y is likely to be low
    samples = mu + sigma * z
    
    # Adaptive exploration bonus: add extra weight to high-uncertainty regions
    # This creates a super-linear preference for uncertain areas
    sigma_max = np.max(sigma)
    if sigma_max > 1e-9:
        s_norm = sigma / sigma_max
        # Extra exploration bonus that grows with uncertainty
        extra_beta = 0.3 * s_norm
    else:
        extra_beta = np.zeros(n)
    
    # Combine: negative sample (exploitation) + adaptive exploration bonus
    # The exploration bonus encourages visiting uncertain regions more aggressively
    scores = -samples + extra_beta * sigma
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
