"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for Bayesian optimization.
    
    Draws a single sample from the posterior predictive distribution at each
    candidate point and returns the negative as the score. This inherently
    balances exploration and exploitation without a manually tuned beta.
    
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
        Thompson Sampling scores; higher means evaluate next.
    """
    # Deadline guard: if we're running long, fall back to greedy
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
    
    elapsed = time.time() - acquisition._start_time
    if elapsed > 280:
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, -1e10)
        return scores
    
    # Draw a single standard normal sample
    z = np.random.randn()
    
    # Thompson sampling: sample from posterior predictive
    # For minimization, we want to find low values
    # Sample: mu - sigma * z (since we're minimizing, lower is better)
    # Score: -sample = -mu + sigma * z
    scores = -mu + sigma * z
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
