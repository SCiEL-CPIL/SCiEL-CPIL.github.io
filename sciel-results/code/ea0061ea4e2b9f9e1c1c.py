"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for Bayesian optimization.
    
    Draws a single sample from the posterior predictive distribution at each
    candidate point and returns the negative of that sample as the score.
    Higher scores mean more likely to be evaluated next (i.e., lower predicted
    objective value).
    
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
    # Use a fixed seed for reproducibility within a single call, but vary it
    # based on f_best to get different samples each iteration
    # Actually, we want randomness to escape local optima, so use a random seed
    # derived from the current state to ensure different samples each time
    
    # Draw a single standard normal sample for Thompson Sampling
    # This is the key structural change: instead of a deterministic acquisition
    # like EI, we sample from the posterior predictive distribution
    
    # For Thompson Sampling in minimization:
    # We sample y ~ N(mu, sigma^2) and want to minimize y
    # So we return -y as the score (higher score = lower predicted value)
    
    # Use a deterministic but state-dependent random seed to ensure
    # different samples at each iteration while being reproducible
    # We'll use np.random with a seed based on f_best and the current time
    
    # Actually, for proper Thompson Sampling, we should use true randomness
    # to explore the posterior. Let's use a global random state that changes
    # each call.
    
    # Draw samples from N(0, 1)
    z = np.random.standard_normal(len(mu))
    
    # Posterior predictive samples: y = mu + sigma * z
    # For minimization, we want to find points where y is likely to be low
    # Score = -y (higher is better for our "evaluate next" criterion)
    samples = mu + sigma * z
    scores = -samples
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
