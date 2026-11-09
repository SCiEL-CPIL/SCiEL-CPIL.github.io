# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling for Bayesian optimization.
    
    This approach draws a sample from the posterior distribution at each candidate
    point and selects the point with the lowest sampled value (highest score, since
    we return negative values). This naturally balances exploration and exploitation.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean at candidate points.
    sigma : array-like, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed function value.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising.
    """
    t_start = time.time()
    deadline = t_start + 280.0  # Safety margin
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    n = len(mu)
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Ensure sigma is positive for numerical stability
    # In GP, sigma should be > 0, but numerical errors can make it 0 or negative
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Thompson Sampling:
    # Draw a sample from the posterior N(mu, sigma^2)
    # For each candidate, sample a potential function value
    # The candidate with the lowest sampled value is the most promising
    # We return the negative of the sample so that 'higher score' means 'better'
    
    # Generate standard normal random variables
    # Using np.random.standard_normal is fast and efficient
    z = np.random.standard_normal(n)
    
    samples = mu + sigma_safe * z
    
    # We want to minimize the function, so the best candidate has the lowest sample.
    # The acquisition score should be higher for better candidates.
    # Therefore, we return -sample.
    scores = -samples
    
    # Ensure all scores are finite
    # If any sample is inf or nan, replace it with a very low score (bad candidate)
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check time (though this operation is very fast)
    if time.time() > deadline:
        # Fallback to a simple deterministic heuristic if time is tight (unlikely)
        # But since we are already done with the calculation, we just return
        pass
    
    return scores
# EVOLVE-BLOCK-END
