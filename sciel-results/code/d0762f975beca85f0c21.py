"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time to track total elapsed time for the whole run
_START_TIME = time.time()
TIME_LIMIT = 330.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Powell 4D.
    
    Structural Change:
    Replaces the static/heuristic beta (based on mean_cost) with a dynamic beta
    based on the current best function value (f_best). This explicitly shifts
    the exploration-exploitation balance:
    - Early (high f_best): Low confidence, low cost penalty (explore).
    - Late (low f_best): High confidence, high cost penalty (exploit cheaply).
    """
    # Deadline Guard: Check if we are close to the time limit.
    # If so, return a simple, fast score to allow the loop to finish gracefully.
    elapsed = time.time() - _START_TIME
    if elapsed > TIME_LIMIT - 5.0:
        # Fallback: Simple Expected Improvement, ignoring cost to save time/logic
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.maximum(ei, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Expected Improvement for minimization: E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Dynamic Beta based on f_best
    # We map f_best to a "Confidence" score C in [0, 1].
    # High f_best -> Low Confidence -> Low Beta (Explore, ignore cost more)
    # Low f_best  -> High Confidence -> High Beta (Exploit, penalize cost heavily)
    
    # Heuristic parameters for the sigmoid:
    # Assume f_best starts high (e.g., 50-100) and drops to 0.
    # We want the transition to happen when f_best is reasonably low, say < 10.
    # C = 1 / (1 + exp(a * (f_best - b)))
    # If f_best >> b, C -> 0.
    # If f_best << b, C -> 1.
    
    # Let's set b=5.0, a=0.5.
    # If f_best = 50, exp(0.5*45) is huge, C ~ 0.
    # If f_best = 0, exp(0.5*-5) = exp(-2.5) ~ 0.08, C ~ 0.92.
    # If f_best = 5, C = 0.5.
    
    a_param = 0.5
    b_param = 5.0
    
    # Clamp f_best for numerical stability in exp
    f_best_clamped = np.clip(f_best, -100, 100)
    
    confidence = 1.0 / (1.0 + np.exp(a_param * (f_best_clamped - b_param)))
    
    # Beta ranges from 0.1 (low confidence) to 1.0 (high confidence)
    beta_min = 0.1
    beta_max = 1.0
    beta = beta_min + confidence * (beta_max - beta_min)
    
    # Compute acquisition score
    # Penalty term: beta * sigma * cost
    # This penalizes uncertain points that are expensive.
    score = ei - beta * sigma_safe * cost
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
