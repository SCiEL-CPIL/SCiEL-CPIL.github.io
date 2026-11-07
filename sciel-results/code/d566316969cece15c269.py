# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Instead of computing a deterministic score like EI or LCB, we sample a single
    function value from the posterior distribution N(mu, sigma^2) for each candidate.
    The candidate with the lowest sampled value is selected. This is equivalent to
    maximizing the score -sampled_value.
    
    This approach is structurally different from EI/LCB blends:
    - It does not require tuning beta or xi.
    - It naturally balances exploration (high sigma regions have high variance in samples)
      and exploitation (low mu regions tend to have lower samples).
    - It avoids the "greedy trap" of EI which can get stuck in local minima if the
      initial points are not diverse enough.
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety check for time
    if time.time() - start_time > 325:
        return -mu

    # Handle edge cases where sigma might be 0 or negative due to numerical issues
    # sigma is standard deviation, so it should be >= 0.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Sample from the posterior: x ~ N(mu, sigma^2)
    # We need to generate random numbers. Since the function is stateless,
    # the randomness comes from the global numpy random state.
    # This is acceptable as the evaluator likely doesn't reset the seed between calls
    # in a way that would make TS deterministic, but even if it did, the stochastic
    # nature is the point. However, to be safe and reproducible per-call if needed,
    # we could use a local generator, but standard np.random is fine for BO.
    
    # Generate standard normal random numbers
    # We use np.random.randn for efficiency
    eps = np.random.randn(mu.shape)
    
    # Sampled values
    sampled_values = mu + sigma_safe * eps
    
    # We want to minimize the objective, so the candidate with the lowest
    # sampled value is the best.
    # The acquisition function returns scores where HIGHER is better.
    # So we return -sampled_values.
    
    scores = -sampled_values
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check time again before returning
    if time.time() - start_time > 328:
        # Fallback to a simple deterministic score if something went wrong
        return -mu - sigma_safe
        
    return scores
# EVOLVE-BLOCK-END
