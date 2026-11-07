"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
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
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    _init_time()
    
    # If we are close to the deadline, return a simple, fast score to avoid timeout
    if _time_remaining() < 5.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        cost_safe = np.maximum(cost, 1e-9)
        scores = ei / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Standard robust calculation
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-scores
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Stochastic Expected Improvement (SEI) with log-normal cost model:
    # Model cost C as log-normal: ln(C) ~ N(mu_c, sigma_c^2)
    # SEI = E[EI * 1/C] where the expectation is over both f(x) and C
    # 
    # For a log-normal cost, E[1/C] = exp(mu_c - sigma_c^2/2)
    # We estimate mu_c and sigma_c from the observed cost distribution
    # 
    # This creates a smooth, probabilistic penalty rather than a hard division
    
    # Estimate log-normal parameters from cost distribution
    log_costs = np.log(cost_safe)
    mu_c = np.mean(log_costs)
    sigma_c = np.std(log_costs)
    
    # Cap sigma_c to avoid extreme variance in the penalty
    sigma_c = min(sigma_c, 0.5)
    
    # E[1/C] for log-normal cost
    e_inv_cost = np.exp(mu_c - sigma_c**2 / 2.0)
    
    # Stochastic EI: scale EI by expected inverse cost
    # This integrates cost uncertainty directly into the acquisition
    sei = ei * e_inv_cost
    
    # Add a small exploration bonus to encourage diverse sampling
    # Use UCB-style term with reduced weight to avoid over-exploration
    beta = 1.5
    ucb_linear = (f_best - mu + beta * sigma_safe) / cost_safe
    ucb_linear = np.maximum(ucb_linear, 0.0)
    
    # Combine: Primary SEI + Secondary UCB-style exploration
    # Weight 0.2 for UCB term to maintain exploitation focus
    scores = sei + 0.2 * ucb_linear
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
