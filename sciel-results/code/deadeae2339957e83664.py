"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_START_TIME = time.time()
_DEADLINE = 295.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are close to the limit, return a fast, safe score
    if time.time() - _START_TIME > _DEADLINE:
        # Return a simple exploitation score to avoid complex calculations
        return -mu / cost

    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) term for minimization
    # We maximize this score.
    # Optimistic view: f_best - (mu - beta * sigma)
    beta = 2.0
    ucb_term = f_best - mu + beta * sigma_safe
    
    # Adaptive blending weight
    # High f_best (early stage) -> High exploration (high w)
    # Low f_best (late stage) -> Low exploration (low w)
    # Map f_best from [0, 2] to [0.1, 0.8]
    w = 0.1 + 0.7 * np.clip(f_best / 2.0, 0.0, 1.0)
    
    # Blend EI and UCB
    score = (1.0 - w) * ei + w * ucb_term
    
    # Cost-aware normalization
    score = score / cost
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
