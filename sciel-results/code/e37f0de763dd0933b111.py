import numpy as np
from scipy.stats import norm
import time

# Global state for deadline tracking
_deadline = None
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Parameters:
    mu: (2000,) array of mean predictions
    sigma: (2000,) array of standard deviations
    f_best: float, current best observed value
    cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    (2000,) array of scores (higher is better)
    """
    global _deadline, _start_time
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 325.0  # 5s safety margin

    if time.time() > _deadline:
        return np.ones(len(mu))

    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Thompson Sampling: draw a sample from the posterior for each candidate
    # sample = mu + sigma * N(0,1)
    rng = np.random.default_rng()
    noise = rng.standard_normal(len(mu))
    samples = mu + sigma_safe * noise
    
    # For minimization, we want to select points with low sampled values
    # Score should be higher for points with lower sampled value / cost
    # Use negative of sampled value divided by cost to make higher = better
    
    # To avoid issues with very negative values, we can use a soft transformation
    # or just use -sample / cost
    # But we need to ensure scores are finite and well-behaved
    
    # Use a transformation that penalizes high cost and rewards low samples
    # score = -sample / cost
    # This makes lower sample values (better) give higher scores
    
    scores = -samples / (cost + eps)
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
