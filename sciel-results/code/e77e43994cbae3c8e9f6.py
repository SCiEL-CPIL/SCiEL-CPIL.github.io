"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Cache for expensive computations to avoid recomputation if called multiple times
# Note: The task says "Each function runs on five seeded instances", implying the
# acquisition function is called once per instance, or potentially multiple times
# within an instance's loop. The signature is acquisition(mu, sigma, f_best, cost).
# mu and sigma have shape (2000,).
# We need to return 2000 finite scores.

# Deadline guard
_START_TIME = time.time()
DEADLINE = _START_TIME + 325.0  # 5 seconds safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores for 2000 candidates.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Lowest observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x scaled to [0,1])).
        Since x is in [-2, 2]^2, scaled to [0,1] is (x+2)/4.
        mean(x_scaled) = mean((x+2)/4) = (mean(x)+2)/4.
        cost = 1 + (mean(x)+2)/4.
        However, cost is passed directly, so we just use it.
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Check deadline
    if time.time() > DEADLINE:
        # Return a safe fallback: random or uniform scores
        return np.ones_like(mu, dtype=np.float64) * 1e-10
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-10)
    
    # Lower Confidence Bound (LCB) for minimization
    # We want to minimize mu - kappa * sigma
    # So we maximize (kappa * sigma - mu)
    # kappa controls the exploration-exploitation trade-off
    # Higher kappa means more exploration
    kappa = 2.0
    lcb_score = kappa * sigma - mu
    
    # Cost-aware: divide by cost
    # To avoid division by zero or very small costs, we already clipped cost.
    scores = lcb_score / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative (EI can be negative if mu > f_best, but we want higher scores for better candidates)
    # Actually, EI is always non-negative for minimization when f_best is the current best.
    # If mu < f_best, z > 0, EI > 0.
    # If mu > f_best, z < 0, EI can be small or negative? 
    # For minimization, EI = E[max(f_best - f(x), 0)]. So EI >= 0 always.
    # But numerically, if mu is much larger than f_best, z is very negative, cdf(z) ~ 0, pdf(z) ~ 0, so EI ~ 0.
    # So scores should be >= 0.
    
    return scores
# EVOLVE-BLOCK-END
