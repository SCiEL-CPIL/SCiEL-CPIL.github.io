"""
Cost-aware acquisition function for Bayesian optimization (shekel 4d).
Uses a hybrid Stochastic Expected Improvement (SEI) and UCB approach.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state for deadline tracking
_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    global _START_TIME
    if _START_TIME is None:
        return float('inf')
    # 330s hard limit, use 300s as safety margin for the evaluator's kill switch
    return 300.0 - (time.time() - _START_TIME)

def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores using a hybrid SEI/UCB approach.
    
    Args:
        mu: Mean of the GP prediction (shape: 2000,)
        sigma: Std deviation of the GP prediction (shape: 2000,)
        f_best: Best observed value (scalar)
        cost: Cost of evaluating each point (shape: 2000,)
        
    Returns:
        scores: Acquisition scores (shape: 2000,), higher is better.
    """
    _init_time()
    
    # Deadline Guard: If close to limit, use a fast, simple score
    if _time_remaining() < 5.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        scores = ei / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # --- Main Algorithm: Hybrid SEI (EI) + UCB ---
    
    # Ensure sigma and cost are safe for division
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Stochastic Expected Improvement (EI)
    # EI captures the expected magnitude of improvement.
    # It is "stochastic" in the sense that it integrates over the posterior distribution.
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI (CEI)
    cea = ei / cost_safe
    
    # 2. UCB-style Exploration Term
    # UCB encourages exploring regions with high uncertainty.
    # For minimization, we want low (mu - beta * sigma).
    # Score contribution: (f_best - mu + beta * sigma) / cost
    beta = 2.0
    ucb_linear = (f_best - mu + beta * sigma_safe) / cost_safe
    ucb_linear = np.maximum(ucb_linear, 0.0)
    
    # 3. Combine
    # The "Stochastic" nature of EI provides a robust exploitation signal.
    # The UCB term provides a strong exploration signal.
    # We weight them to balance the two.
    # Inspiration 1 found 0.3 weight for UCB to be effective.
    w_ucb = 0.3
    scores = cea + w_ucb * ucb_linear
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
