import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_TIME_LIMIT = 295.0  # Leave 5s margin before 300s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB).
    
    UCB is a structural alternative to EI that balances exploitation (low mu)
    and exploration (high sigma) more robustly in multimodal landscapes.
    It can be viewed as a PES approximation where the entropy reduction is
    driven by the uncertainty of the function value relative to the best.
    
    Args:
        mu: Array of mean predictions (2000,).
        sigma: Array of standard deviation predictions (2000,).
        f_best: The best (lowest) function value observed so far.
        
    Returns:
        Array of scores (2000,). Higher scores indicate better candidates.
    """
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-8)
        return -(mu + 0.5 * sigma_safe)
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # UCB for minimization:
    # We want to minimize f(x).
    # UCB(x) = mu(x) - beta * sigma(x)
    # We want to maximize the "score", so we return -UCB(x) or similar.
    # Score = -mu(x) + beta * sigma(x)
    
    # Choose beta. 
    # A fixed beta of 1.0 or 2.0 is common.
    # Since sigma is in objective units, and Weierstrass values are ~0-1,
    # sigma might be small. 
    # Let's try beta = 1.5.
    
    beta = 1.5
    score = -mu + beta * sigma_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
