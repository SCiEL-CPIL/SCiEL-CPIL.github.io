import numpy as np
import time

# Global time tracking
_start_time = time.time()
TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Hypar Improvement (EHI) acquisition function.
    
    For minimization, we maximize the expected hyperbolic tangent of the
    normalized improvement. This explicitly models the trade-off between
    function value and evaluation cost, avoiding UCB's over-exploration
    of high-cost, high-uncertainty regions.
    
    Score = E[tanh((f_best - f(x)) / (cost * scale))]
    
    where f(x) ~ N(mu, sigma^2). We use the approximation:
    E[tanh(Z)] ≈ tanh(m) * exp(-0.5 * v)
    where m = (f_best - mu) / (cost * scale)
    and v = (sigma / (cost * scale))^2
    """
    # Check time
    current_time = time.time()
    if current_time - _start_time > TIME_LIMIT - 5:
        # Emergency fallback: favor low mu and low cost
        cost_safe = np.maximum(cost, 1e-8)
        return -mu / cost_safe

    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)

    # Scale parameter for normalization
    # This controls the sensitivity of the tanh to improvement magnitude
    scale = 1.0
    
    # Compute normalized mean and standard deviation
    mean_z = (f_best - mu) / (cost * scale)
    std_z = sigma / (cost * scale)
    
    # Hybrid EHI + UCB-like exploration
    # Exploitation: tanh(mean_z) rewards low mu (good improvement)
    # Exploration: beta * tanh(std_z) rewards uncertainty, but saturates
    # Both terms are naturally dampened by high cost via the normalization
    beta = 0.2
    
    tanh_mean = np.tanh(mean_z)
    tanh_std = np.tanh(std_z)
    
    # Score: higher is better
    score = tanh_mean + beta * tanh_std
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    return score
