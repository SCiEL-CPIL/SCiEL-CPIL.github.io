"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Instead of computing a deterministic score like EI or UCB, we sample 
    a single value from the posterior distribution N(mu, sigma^2) for each 
    candidate. We then select the candidate with the lowest sampled value.
    
    This approach naturally balances exploration and exploitation:
    - In high-uncertainty regions, the sampled value can be very low (good) 
      or very high (bad) due to the large sigma, encouraging exploration.
    - In low-uncertainty regions, the sampled value is close to the mean, 
      favoring exploitation of known good areas.
    
    Since the loop selects the candidate with the HIGHEST score, we return 
    the negative of the sampled value.
    
    mu: (N,) posterior mean
    sigma: (N,) posterior std
    f_best: best observed value so far (not used in pure TS, but kept for interface)
    
    Returns: (N,) scores, higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid sampling errors
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Generate a single random standard normal sample for all candidates
    # This ensures the randomness is consistent across the candidate set for this step
    # (i.e., we are effectively sampling a single function instance from the GP)
    z = np.random.standard_normal(mu.shape[0])
    
    # Sample from the posterior: f ~ N(mu, sigma^2)
    samples = mu + sigma_safe * z
    
    # We want to minimize the function, so we return negative samples
    # so that the "highest score" corresponds to the "lowest function value"
    score = -samples
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
