import numpy as np
import time

# Global timer for safety
_start_time = time.time()
_DEADLINE = 320.0  # 10s margin before 330s kill

# State tracking for adaptive parameters
_call_count = 0
TOTAL_STEPS = 25  # 30 total evals - 5 initial = 25 acquisition calls

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses Time-Adaptive Upper Confidence Bound (UCB) for minimization.
    Score = -mu + beta(t) * sigma
    
    The beta parameter is adapted over the course of the optimization:
    - Early steps: High beta (exploration) to cover the domain and find the global basin.
    - Late steps: Low beta (exploitation) to refine the best candidate.
    """
    global _call_count
    
    # Check time budget
    if time.time() - _start_time > _DEADLINE:
        return np.zeros_like(mu)

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-10)
    
    # Calculate adaptive beta
    # t ranges from 0 to TOTAL_STEPS - 1
    t = _call_count
    _call_count += 1
    
    # Linear decay from 4.0 to 0.5
    beta_start = 4.0
    beta_end = 0.5
    if TOTAL_STEPS > 1:
        progress = t / (TOTAL_STEPS - 1)
    else:
        progress = 0.0
    beta = beta_start + (beta_end - beta_start) * progress
    
    # UCB score for minimization:
    # We want to find points with low mu (good) and high sigma (uncertain)
    # score = -mu + beta * sigma
    score = -mu + beta * sigma
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
