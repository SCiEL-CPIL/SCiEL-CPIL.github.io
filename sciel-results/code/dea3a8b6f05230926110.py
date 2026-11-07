import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Args:
        mu: (2000,) array of predicted means in objective units.
        sigma: (2000,) array of predicted standard deviations in objective units.
        f_best: float, the lowest observed value.
        
    Returns:
        (2000,) array of scores. Higher scores indicate higher priority for evaluation.
    """
    # Deadline guard (unlikely to be needed, but required by directive)
    start_time = time.time()
    deadline = start_time + 0.1  # 100ms safety margin, computation is <1ms
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: draw a sample from the posterior for each candidate
    z = np.random.standard_normal(len(mu))
    samples = mu + sigma_safe * z
    
    # We are minimizing. The best candidate is the one with the lowest sample.
    # We return scores such that higher is better.
    scores = -samples
    
    # Check deadline (just in case)
    if time.time() > deadline:
        # Fallback: return deterministic scores based on mu only (greedy)
        # This should never happen for N=2000
        scores = -mu
        
    return scores
