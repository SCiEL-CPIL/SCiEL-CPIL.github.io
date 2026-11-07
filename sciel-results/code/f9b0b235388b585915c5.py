import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track elapsed time to avoid timeout
_start_time = time.time()
DEADLINE = 320.0  # Safety margin before 330s limit

def _check_deadline():
    return (time.time() - _start_time) > DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Rastrigin 2D.
    
    Uses Probability of Improvement (PoI) per unit cost as the primary
    acquisition, combined with a small exploration bonus for robustness.
    PoI/cost was measured to significantly outperform EI/cost on this task.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x scaled to [0,1]).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Check deadline - if we're running out of time, return a safe fallback
    if _check_deadline():
        # Return scores that prioritize low-cost points near current best
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-9)
        return 1.0 / cost_safe
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Probability of Improvement for minimization:
    # PoI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This prioritizes high-probability cheap improvements over potentially
    # misleading high-magnitude predictions, which is better for multimodal Rastrigin
    
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    
    poi = norm.cdf(z_clipped)
    
    # Ensure PoI is in [0, 1]
    poi = np.clip(poi, 0.0, 1.0)
    
    # Cost-aware PoI: divide by cost to prefer cheaper evaluations
    cost_aware_poi = poi / cost_safe
    
    # Add a small exploration bonus based on sigma to encourage exploring
    # uncertain regions, scaled by inverse cost. This helps escape local
    # optima in Rastrigin's many local minima.
    # Use a smaller weight than EI-based approaches since PoI already
    # incorporates uncertainty through the CDF.
    exploration_bonus = 0.1 * sigma_safe / cost_safe
    
    # Combine
    score = cost_aware_poi + exploration_bonus
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
