"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track start time and ensure we return before 330s
_start_time = time.time()
DEADLINE = 325.0  # Safety margin of 5 seconds

def _check_deadline():
    return (time.time() - _start_time) < DEADLINE

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Check deadline
    if not _check_deadline():
        # Return a safe default: prioritize low mu / high uncertainty
        # Use a simple UCB-like score
        eps = 1e-9
        sigma_safe = np.maximum(sigma, eps)
        return -mu + 0.1 * sigma_safe
    
    # Clamp sigma to avoid division issues
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Clamp cost to avoid division by zero
    cost_safe = np.maximum(cost, 0.1)
    
    # Compute z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Cost-normalized EI
    ei_per_cost = ei / cost_safe
    
    # Add a small exploration term: UCB-like bonus for high uncertainty
    # Scale by inverse cost to be cost-aware
    beta = 0.05  # Small exploration weight
    ucb_bonus = beta * sigma_safe / cost_safe
    
    # Combine: primarily EI/cost, with small exploration bonus
    score = ei_per_cost + ucb_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Clip to reasonable range to avoid extreme values
    score = np.clip(score, -1e6, 1e6)
    
    return score
# EVOLVE-BLOCK-END
