"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 320.0  # Safety margin for 330s limit

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for minimization.
    Returns scores where higher is better.
    
    Strategy:
    Sample a single function value for each candidate from the Gaussian posterior:
    f_sample = mu + sigma * z, where z ~ N(0, 1).
    
    For minimization, we want to pick the candidate with the lowest sampled value.
    The score interface requires "higher meaning evaluate next".
    Therefore, we return scores = -f_sample.
    The candidate with the highest score will have the lowest sampled value.
    
    This approach naturally balances exploration (high sigma leads to large variance,
    potentially sampling very low values) and exploitation (low mu leads to lower
    expected samples).
    """
    # Check deadline
    if time.time() - _start_time > _TIMEOUT:
        # Return a safe fallback. In a timeout scenario, we just return exploitation.
        # Since we can't be sure of the state, return -mu (greedy exploitation).
        # This ensures we return finite numbers.
        return -mu

    # Ensure sigma is non-negative and finite
    sigma = np.asarray(sigma)
    mu = np.asarray(mu)
    
    # Clip sigma to avoid negative values or NaNs if any slipped through
    sigma_safe = np.clip(sigma, 1e-9, None)
    
    # Draw random normal samples
    # shape matches mu and sigma
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Sample function values
    # f_sample = mu + sigma * z
    f_sample = mu + sigma_safe * z
    
    # For minimization, lower f_sample is better.
    # Score should be higher for better candidates.
    # So score = -f_sample.
    scores = -f_sample
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
