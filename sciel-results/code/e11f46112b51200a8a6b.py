"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute constants for speed
_EPS = 1e-9

def _ackley_2d(x):
    """Ackley 2D function (standard, minimum at 0,0)."""
    x1 = x[:, 0]
    x2 = x[:, 1]
    sum_sq = x1**2 + x2**2
    sum_cos = np.cos(2.0 * np.pi * x1) + np.cos(2.0 * np.pi * x2)
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * sum_sq)) - np.exp(0.5 * sum_cos) + np.e + 20.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: deterministic, cost-normalized LCB
    with adaptive beta that shifts from exploration to exploitation
    as budget depletes.
    
    mu, sigma: (2000,) arrays of GP mean and std in objective units
    f_best: scalar, lowest observed value
    cost: (2000,) array of evaluation costs
    
    Returns: (2000,) finite scores, higher = prefer to evaluate next
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)
    
    # Adaptive beta: starts high for exploration, decreases as f_best approaches 0
    # f_best for Ackley 2D ranges from ~20+ down to 0
    # Use a smooth transition: beta is high when f_best is large, low when f_best is small
    f_norm = min(max(f_best / 40.0, 0.0), 1.0)
    
    # Beta transitions from ~2.5 (exploration) to ~0.5 (exploitation)
    # Using a smooth function: beta = 0.5 + 2.0 * f_norm
    # This gives beta in [0.5, 2.5]
    beta = 0.5 + 2.0 * f_norm
    
    # Lower Confidence Bound for minimization:
    # LCB = mu - beta * sigma
    # We want to MINIMIZE LCB, so higher score = lower LCB
    lcb = mu - beta * sigma
    
    # Cost normalization: divide by cost to get per-unit-cost value
    # Lower LCB per cost is better
    # Score = -LCB / cost (negate so higher is better)
    cost_safe = np.maximum(cost, _EPS)
    score = -lcb / cost_safe
    
    # Check deadline
    if time.time() > deadline:
        # Return best result found so far
        pass
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
