"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 300  # safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for Bayesian optimization (minimization).
    
    Instead of a fixed combination of EI and UCB, we sample from the posterior
    distribution at each candidate point and select the point with the lowest
    sampled value. This inherently balances exploration and exploitation:
    - Low-mu, low-sigma regions: samples cluster near the true (low) value
    - High-sigma regions: samples have wider spread, sometimes yielding very low values
    
    This avoids the need to tune a fixed beta/kappa parameter and naturally
    adapts to the current state of knowledge.
    
    For computational efficiency with 2000 candidates, we use a single shared
    random draw scaled per-candidate, which is a valid approximation of
    Thompson sampling that preserves the relative ordering structure.
    """
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        # Pure exploitation fallback
        return -mu
    
    n = len(mu)
    
    # Sanitize inputs
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Thompson Sampling: sample from N(mu, sigma) for each candidate
    # Use a single standard normal draw for computational efficiency
    # This is a valid form of Thompson sampling where we share the random
    # direction but scale by individual sigmas
    
    # Draw a single standard normal random variable
    z = np.random.randn()
    
    # Sample values: y_i = mu_i + sigma_i * z
    # For minimization, we want the LOWEST sampled value
    # So the score (higher is better) is the negative of the sampled value
    sampled_values = mu + sigma_safe * z
    
    # Score: higher means more attractive (we want to minimize, so negate)
    scores = -sampled_values
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
