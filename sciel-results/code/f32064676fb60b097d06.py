"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global time tracking to respect the 330s wall-clock limit
_start_time = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 10.0  # seconds

def acquisition(mu, sigma, f_best, cost):
    """
    Compute acquisition scores for 2000 candidates.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, lowest observed value.
        cost: (2000,) array of evaluation costs.
    
    Returns:
        (2000,) array of finite scores (higher = evaluate next).
    """
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT - SAFETY_MARGIN:
        # Return a safe fallback: just pick the one with lowest mu / cost
        # to avoid crashing, though the run is likely over.
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-9)
        mu_arr = np.asarray(mu, dtype=np.float64)
        return -mu_arr / cost_safe

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Strategy: Adaptive Lower Confidence Bound (LCB) per unit cost.
    # The best performing inspiration (score 2.3655) used an adaptive beta LCB.
    # LCB for minimization: we want to minimize mu - beta * sigma.
    # Maximizing the acquisition function means maximizing -mu + beta * sigma.
    
    # Ackley 2D max is approx 29. We use a scale of 25.0 for normalization.
    f_scale = 25.0
    
    # Adaptive beta: high when f_best is high (early, explore), low when f_best is low (late, exploit)
    # beta = beta_min + (beta_max - beta_min) * (f_best / f_scale)
    beta_min = 0.5
    beta_max = 3.0
    
    # Ensure f_best is positive for the ratio, though it should be.
    # Clip to [0, 1] to keep beta within [beta_min, beta_max]
    f_ratio = np.clip(f_best / f_scale, 0.0, 1.0)
    beta = beta_min + (beta_max - beta_min) * f_ratio
    
    # LCB score for minimization: -mu + beta * sigma
    # This encourages exploring areas with high uncertainty (sigma) when f_best is still high.
    lcb_score = -mu + beta * sigma_safe
    
    # Cost adjustment: divide by cost to get "improvement per unit cost"
    score = lcb_score / cost_safe
    
    # Ensure finiteness
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return score
# EVOLVE-BLOCK-END
