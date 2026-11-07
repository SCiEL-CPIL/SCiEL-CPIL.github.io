# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 325s as safety margin
_DEADLINE = 325.0
_START_TIME = time.time()

def _check_deadline():
    """Return True if we should abort and return current best."""
    return (time.time() - _START_TIME) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    Strategy:
    - Compute standard Expected Improvement: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    - Apply a cost penalty that is proportional to the uncertainty (sigma).
      This penalizes expensive evaluations more when the model is uncertain,
      but allows expensive evaluations when the expected improvement is high
      and the model is confident (low sigma).
    - The penalty is scaled by a hyperparameter to balance exploration cost.
    
    This differs from LCB/cost because it doesn't suppress the entire score
    for expensive points, but rather subtracts a cost term that depends on
    how "risky" (uncertain) the evaluation is.
    """
    # Check deadline first
    if _check_deadline():
        # Return a neutral score if we're running out of time
        return np.zeros_like(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Clip sigma to avoid numerical issues in z calculation
    sigma = np.clip(sigma, 1e-6, 1e6)
    cost = np.clip(cost, 0.5, 100.0)
    
    # Calculate Expected Improvement
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma
    
    # Handle potential overflow in exp for large negative z
    # For z < -10, Phi(z) ~ 0, phi(z) ~ 0
    # For z > 10, Phi(z) ~ 1, phi(z) ~ 0
    # We can compute directly for stability in normal ranges
    
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_vals + sigma * pdf_vals
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost penalty:
    # We want to penalize expensive evaluations.
    # The penalty should be higher when uncertainty is high (exploratory) 
    # and lower when the point is a strong exploitation candidate (low sigma, high EI).
    # 
    # Simple approach: subtract a fixed cost penalty scaled by a factor.
    # Better approach: subtract cost * sigma * alpha.
    # If sigma is small (confident), the penalty is small, allowing expensive points 
    # if EI is high. If sigma is large (uncertain), the penalty is large, 
    # discouraging expensive exploratory moves unless EI is very high.
    
    # Let's use a penalty proportional to (cost - 1) * sigma * alpha
    # cost is in [1, 2]. (cost - 1) is in [0, 1].
    # alpha controls the strength of the cost penalty.
    
    # Adaptive alpha: 
    # Early on (f_best far from min), we might want to be more aggressive with cost.
    # Late on, we want to be more careful.
    # However, a fixed alpha is often robust. Let's start with a moderate value.
    
    # The scale of EI depends on the problem. For Styblinski-Tang, values are ~ -78 to 0.
    # Typical EI might be in the range of 0 to 10.
    # Cost penalty: (cost - 1) is ~ 0 to 1. sigma is ~ 1 to 10.
    # So penalty is ~ 0 to 10 * alpha.
    # If alpha = 0.1, penalty is ~ 0 to 1.
    # If alpha = 0.5, penalty is ~ 0 to 5.
    
    # Let's use an adaptive alpha that decreases as we get closer to the minimum.
    # dist_to_min = f_best - (-78.3323)
    # When dist is large, we are far from min, EI values might be larger in absolute terms?
    # Actually, EI is bounded by f_best - min_possible.
    
    # Let's stick to a simple, robust penalty:
    # penalty = (cost - 1.0) * sigma * 0.2
    
    # This means:
    # - Cheap points (cost~1): penalty ~ 0
    # - Expensive points (cost~2): penalty ~ sigma * 0.2
    # - If sigma is 1, penalty ~ 0.2
    # - If sigma is 10, penalty ~ 2.0
    
    # This seems reasonable. It discourages expensive exploratory moves.
    
    penalty = (cost - 1.0) * sigma * 0.2
    
    score = ei - penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
