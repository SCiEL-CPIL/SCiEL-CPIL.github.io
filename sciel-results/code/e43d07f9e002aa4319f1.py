"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global timer for safety
_t0 = time.time()
_DEADLINE = 320.0


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    For each candidate, draw a single sample from the posterior:
        theta = mu + sigma * Z
    where Z is standard normal noise.
    Score = -theta (we want to minimize, so lower theta is better).
    
    This implicitly balances exploration and exploitation:
    - High sigma regions get more random variation (exploration)
    - Low sigma regions are dominated by mu (exploitation)
    
    mu and sigma are in objective units (lower is better for the objective).
    We want to maximize the acquisition score.
    """
    # Check deadline
    if time.time() - _t0 > _DEADLINE:
        # Return a score that favors low mu (exploitation) if we're out of time
        return -mu

    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Draw a single standard normal random vector for all candidates
    # This ensures consistent random perturbation across all candidates
    Z = np.random.standard_normal(mu.shape)
    
    # Thompson Sampling: sample from posterior
    theta = mu + sigma_safe * Z
    
    # Score: we want to minimize theta, so higher score = lower theta
    score = -theta
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
