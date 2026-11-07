"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state to track time for safety, though acquisition is fast
_start_time = time.time()
_DEADLINE = 300.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Instead of computing a deterministic score like EI or LCB, we sample a single
    value from the posterior distribution N(mu, sigma^2) for each candidate point.
    We return the negative of this sample, as we want to maximize the score
    (i.e., find the lowest function value).
    
    This approach naturally balances exploration and exploitation:
    - High mu (good mean) leads to high score.
    - High sigma (high uncertainty) leads to a wider distribution, increasing the
      chance of sampling a very low value (high score) even if the mean is mediocre.
    """
    # Check time limit (though this should rarely trigger for 2000 points)
    if time.time() > _DEADLINE - 10:
        # Fallback to simple exploitation if running out of time
        # Return negative mu to prioritize best known mean
        return -mu

    # Sample from N(mu, sigma^2)
    # sigma is the standard deviation
    # We want to find the minimum, so we maximize -f(x)
    # Thus score = -sample
    samples = mu + sigma * np.random.standard_normal(len(mu))
    scores = -samples
    
    # Ensure finite scores
    # If sigma is very small or mu is extreme, samples should still be finite
    # but let's be safe
    if not np.all(np.isfinite(scores)):
        # Fallback to deterministic LCB if numerical issues arise
        z = (f_best - mu) / (sigma + 1e-9)
        scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        
    return scores
# EVOLVE-BLOCK-END
