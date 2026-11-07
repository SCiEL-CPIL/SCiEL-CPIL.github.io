# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    Combines Expected Improvement (EI) for exploitation with an adaptive
    exploration bonus (UCB-like) that decays as f_best improves.
    """
    start_time = time.time()
    deadline = 320.0  # Safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety checks: ensure finite values and avoid division by zero
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    sigma = np.maximum(sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Adaptive exploration bonus (UCB-like term)
    # Starts higher when f_best is large (early), decreases as f_best -> 0 (late)
    max_f = 25.0
    ratio = np.clip(f_best / max_f, 0.0, 1.0)
    beta = 0.3 + 1.2 * np.sqrt(ratio)
    
    # UCB component for minimization: -mu + beta * sigma
    ucb = -mu + beta * sigma
    
    # Combine EI and UCB with a fixed weight
    # EI dominates exploitation, UCB provides exploration
    score = ei + 0.5 * ucb
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Deadline guard: if computation took too long, return a safe fallback
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: return EI only (fastest path)
        return ei
    
    return score
# EVOLVE-BLOCK-END
