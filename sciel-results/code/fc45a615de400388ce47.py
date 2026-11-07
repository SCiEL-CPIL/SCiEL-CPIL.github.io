import numpy as np
import time
from scipy.stats import norm

# Global variable to track start time for deadline guard
_start_time = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 10.0  # seconds

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    Blends Expected Improvement (EI) and UCB-like exploration, scaled by cost.
    Score = (alpha * EI + (1 - alpha) * (beta * sigma - mu)) / cost
    """
    # Deadline guard
    current_time = time.time()
    if current_time - _start_time > (TIME_LIMIT - SAFETY_MARGIN):
        return np.zeros_like(mu)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    f_best = float(f_best)

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)

    # Expected Improvement for minimization
    # Improvement is (f_best - mu) when mu < f_best
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # UCB-like exploration term for minimization
    # We want to maximize (beta * sigma - mu)
    # Use a moderate beta for exploration
    beta = 1.5
    ucb_term = beta * sigma - mu

    # Blend EI and UCB terms
    # EI is already normalized to some extent, but UCB can be larger
    # Let's use a weighted average. alpha controls the mix.
    # Higher alpha means more exploitation (EI), lower alpha means more exploration (UCB)
    alpha = 0.7
    
    # Normalize the terms to have similar scales before blending
    # EI is roughly in the range [0, f_best - min(mu)]
    # UCB is roughly in the range [-max(mu) + beta*max(sigma), -min(mu) + beta*max(sigma)]
    
    # A simple blend
    blended_score = alpha * ei + (1 - alpha) * ucb_term

    # Cost-aware scaling
    # Higher score is better.
    # If cost is high, we penalize the score.
    scores = blended_score / cost

    # Handle potential numerical issues (inf/nan)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scores
