"""Cost-aware acquisition function for Bayesian optimization (4 test functions)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function for cost-aware Bayesian optimization.

    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviations
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - evaluation costs for each candidate

    Returns:
    - array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    t_start = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Safety: clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-scores
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Lower Confidence Bound (LCB) - exploration-focused
    # For minimization, LCB = mu - kappa * sigma
    # Higher LCB score means more promising for exploration
    kappa = 2.0
    lcb = mu - kappa * sigma_safe
    # Convert to "improvement" form: lower lcb is better, so score = -lcb
    lcb_score = -lcb
    
    # Normalize EI and LCB for combination
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    lcb_min = np.min(lcb_score)
    lcb_max = np.max(lcb_score)
    lcb_range = lcb_max - lcb_min if lcb_max > lcb_min else 1.0
    
    ei_norm = ei / ei_max
    lcb_norm = (lcb_score - lcb_min) / lcb_range
    
    # Combine: weighted blend of EI and LCB
    w_ei = 0.6
    w_lcb = 0.4
    combined = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Cost-aware: divide by cost (higher cost = lower score)
    # Use a softer penalty: cost^0.5 to balance exploration vs cost
    cost_penalty = np.sqrt(cost_safe)
    scores = combined / cost_penalty
    
    # Add a small exploration bonus for high uncertainty regions
    # This helps escape local optima
    uncertainty_bonus = 0.1 * (sigma_safe / (np.max(sigma_safe) + 1e-9))
    scores += uncertainty_bonus / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    elapsed = time.time() - t_start
    if elapsed > 320.0:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
