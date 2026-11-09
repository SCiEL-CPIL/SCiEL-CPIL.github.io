import numpy as np
import time
from scipy.stats import norm

# Track start time for deadline guard
_start_time = time.time()
_DEADLINE_MARGIN = 10  # seconds to keep before the 330s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using multi-sample Thompson Sampling with EI anchor
    for minimization of Beale 2D.
    
    Parameters:
    mu (np.array): Mean of the GP posterior, shape (2000,)
    sigma (np.array): Standard deviation of the GP posterior, shape (2000,)
    f_best (float): Best observed function value so far.
    
    Returns:
    np.array: Scores for the 2000 candidates, shape (2000,). Higher is better.
    """
    # Deadline guard
    if time.time() - _start_time > 330 - _DEADLINE_MARGIN:
        return np.zeros_like(mu)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    n = len(mu)
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Ensure sigma is non-negative and safe
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Multi-sample Thompson Sampling: draw K samples, take the minimum
    # This optimizes a lower quantile of the posterior, giving more robust exploration
    K = 5
    samples = mu[None, :] + sigma_safe[None, :] * np.random.standard_normal((K, n))
    ts_score = -np.min(samples, axis=0)  # negative of min sample = higher is better
    
    # Compute Expected Improvement as a deterministic anchor
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    improvement = f_best - mu
    z = improvement / sigma_safe
    z = np.clip(z, -10, 10)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(improvement > 0, ei, 0.0)
    
    # Combine: primarily multi-sample TS, with a small EI anchor for stability
    # The EI term provides a deterministic gradient toward known good regions
    # while TS provides stochastic exploration
    alpha = 0.3  # weight for EI anchor
    scores = alpha * ei + (1.0 - alpha) * ts_score
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
