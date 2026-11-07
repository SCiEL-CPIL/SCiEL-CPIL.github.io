"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a hybrid of Lower Confidence Bound (LCB)
    and Probability of Improvement (PI) beyond a threshold.
    
    The LCB term -(mu - kappa * sigma) / cost provides a smooth surrogate
    that balances exploitation (low mu) and exploration (high sigma) while
    penalizing expensive evaluations. The PI term norm.cdf((f_best - epsilon - mu) / sigma) / cost
    focuses on high-confidence improvements beyond a threshold.
    
    We blend these two signals: the LCB gives a continuous, differentiable
    landscape that avoids the step-like behavior of pure PI, while the PI
    term sharpens focus on likely improvements.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return simple exploitation-based score
        cost_safe = np.maximum(cost, 1e-9)
        score = -mu / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # LCB term: -(mu - kappa * sigma) / cost
    # For minimization, we want to minimize mu + kappa*sigma, so we negate it.
    # kappa controls exploration: higher kappa -> more exploration.
    kappa = 1.0
    lcb_score = -(mu - kappa * sigma_safe) / cost_safe
    
    # PI term: P(f(x) < f_best - epsilon) / cost
    epsilon = 0.01
    z = (f_best - epsilon - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    pi = norm.cdf(z)
    pi_score = pi / cost_safe
    
    # Blend: weight LCB more for smooth exploration, PI for exploitation focus
    # The LCB is in objective units (can be large), PI is in [0,1].
    # Normalize LCB to a comparable scale using median sigma as a reference.
    median_sigma = np.median(sigma_safe)
    # Scale LCB to be roughly in [0, 1] range by dividing by a characteristic scale
    # The typical magnitude of (mu - kappa*sigma) is on the order of f_best magnitude.
    # Use a soft normalization: divide by (|f_best| + 1) to keep it bounded.
    scale = max(abs(f_best), 1.0) + 1.0
    lcb_normalized = lcb_score / scale
    
    # Blend: 50% LCB (normalized) + 50% PI
    # This gives a smooth surrogate that has both exploration and exploitation properties
    score = 0.5 * lcb_normalized + 0.5 * pi_score
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
