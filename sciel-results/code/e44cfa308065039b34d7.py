"""Cost-aware acquisition for rastrigin 2d BO loop."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
_T0 = time.time()
_DEADLINE = 325.0  # safety margin of 5s before the 330s kill


def _deadline_exceeded():
    return (time.time() - _T0) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of rastrigin 2d.
    
    Combines Cost-Aware Lower Confidence Bound (LCB) with Expected Improvement (EI)
    using adaptive weighting. LCB provides strong exploration via the beta*sigma term,
    while EI gives exploitation signal near the optimum. Both are normalized by cost.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're running out of time, return a simple fallback
    if _deadline_exceeded():
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei / (cost + 1e-9)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma and cost are positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-Aware LCB: score = (beta * sigma - mu) / cost
    # beta controls exploration; use a value that balances exploration/exploitation
    # For rastrigin with its many local minima, we want aggressive exploration early
    # and more exploitation as f_best approaches 0
    if f_best < 0.5:
        beta = 1.5
    elif f_best < 2.0:
        beta = 2.0
    elif f_best < 5.0:
        beta = 2.5
    else:
        beta = 3.0
    
    lcb_score = (beta * sigma_safe - mu) / cost_safe
    
    # Normalize EI by cost
    ei_score = ei / cost_safe
    
    # Normalize LCB and EI to comparable scales
    # Use the median of positive values for normalization
    lcb_positive = lcb_score[lcb_score > 0]
    ei_positive = ei_score[ei_score > 0]
    
    if len(lcb_positive) > 0:
        lcb_median = np.median(lcb_positive)
    else:
        lcb_median = 1.0
    lcb_median = max(lcb_median, 1e-6)
    
    if len(ei_positive) > 0:
        ei_median = np.median(ei_positive)
    else:
        ei_median = 1.0
    ei_median = max(ei_median, 1e-6)
    
    lcb_normalized = lcb_score / lcb_median
    ei_normalized = ei_score / ei_median
    
    # Adaptive weighting: more LCB when f_best is large (exploration needed),
    # more EI when f_best is small (exploitation)
    if f_best < 0.5:
        lcb_weight = 0.4
        ei_weight = 0.6
    elif f_best < 2.0:
        lcb_weight = 0.5
        ei_weight = 0.5
    elif f_best < 5.0:
        lcb_weight = 0.6
        ei_weight = 0.4
    else:
        lcb_weight = 0.7
        ei_weight = 0.3
    
    score = lcb_weight * lcb_normalized + ei_weight * ei_normalized
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
