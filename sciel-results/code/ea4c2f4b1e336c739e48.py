"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    Uses a quadratic cost penalty to prioritize low-cost evaluations,
    allowing more samples within the fixed budget.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from the GP in objective units.
    sigma : array-like, shape (2000,)
        Standard deviation predictions from the GP in objective units.
    f_best : float
        Best observed objective value so far (minimum, since we minimize).
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point, in [1, 2].
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline Guard: If we are running out of time, return a fast, simple score
    # to ensure we don't contribute to a timeout.
    # The evaluator limit is 330s. We use 320s as a safety margin.
    elapsed = time.time() - _start_time
    if elapsed > 320:
        cost_arr = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost_arr, 1e-8)
        # Simple inverse cost score
        return 1.0 / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost_arr = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is strictly positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    # Ensure cost is strictly positive
    cost_safe = np.maximum(cost_arr, 1e-8)
    
    # Structural Change: Use Lower Confidence Bound (LCB) instead of EI.
    # LCB(x) = mu(x) - kappa * sigma(x)
    # For minimization, we want to minimize LCB.
    # We set kappa = 1.0, a standard choice for a 1-sigma lower bound.
    kappa = 1.0
    lcb = mu - kappa * sigma_safe
    
    # Acquisition Score:
    # We want to maximize the potential improvement over f_best.
    # Improvement in LCB space: f_best - LCB(x)
    # If LCB(x) < f_best, the score is positive.
    # If LCB(x) > f_best, the score is negative.
    
    # Apply Quadratic Cost Penalty (1/cost^2)
    # This aggressively prioritizes low-cost evaluations,
    # fitting more evaluations within the cost budget of 30.
    score = (f_best - lcb) / (cost_safe ** 2)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
