"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290  # 300s limit, keep 10s margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using normalized EI and UCB.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return np.where(np.isfinite(ei / cost), ei / cost, 0.0)

    # Ensure arrays are float64 for precision
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: clip sigma to avoid division by zero or negative values
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Calculate UCB (Upper Confidence Bound) for exploration
    # Use a slightly higher beta to encourage exploration in early stages
    # Beta is fixed at 1.0 for a stronger exploration signal, normalized later
    beta = 1.0
    ucb = -mu + beta * sigma_safe
    
    # Normalize EI and UCB to [0, 1] to make them comparable
    # Using max value for normalization, with a floor to avoid division by zero
    ei_max = np.max(ei)
    ucb_max = np.max(ucb)
    
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
        
    if ucb_max > 1e-12:
        ucb_norm = ucb / ucb_max
    else:
        ucb_norm = np.zeros_like(ucb)
        
    # Adaptive weight: 
    # If f_best is close to 0, exploit more (higher weight for EI).
    # If f_best is far from 0, explore more (lower weight for EI, higher for UCB).
    # f_best is the minimum observed value. Rosenbrock min is 0.
    # Scale f_best to determine weight.
    # If f_best < 1.0, we are close to optimum, use w=0.95
    # If f_best > 10.0, we are far, use w=0.7
    # Linear interpolation between 1.0 and 10.0
    w = 0.95
    if f_best > 10.0:
        w = 0.7
    elif f_best > 1.0:
        # Interpolate
        t = (f_best - 1.0) / 9.0
        w = 0.95 - t * (0.95 - 0.7)
        
    # Combine normalized scores
    combined = w * ei_norm + (1 - w) * ucb_norm
    
    # Apply cost penalty
    # Use 1/cost to prefer cheaper points
    # To make the cost penalty more significant, we can use exp(-cost) or 1/cost^2
    # 1/cost is mild. Let's try 1/cost^1.5 to amplify cost difference slightly
    cost_penalty = 1.0 / (cost ** 1.5)
    
    score = combined * cost_penalty
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
