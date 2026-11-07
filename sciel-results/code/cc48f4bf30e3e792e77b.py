"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _cost_aware_ucb(mu, sigma, f_best, cost, beta=1.5):
    """
    Cost-aware UCB acquisition with adaptive beta.
    Higher scores are better (we want to minimize the objective).
    """
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # UCB for minimization: lower (mu - beta*sigma) is better
    # Score = (f_best - mu + beta * sigma) / cost
    # Adaptive beta: increase exploration when uncertainty is high
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    # Scale beta based on relative uncertainty
    if sigma_max > 1e-8:
        uncertainty_ratio = sigma_safe / sigma_max
        # Beta ranges from 0.8 to 2.5 based on local uncertainty
        adaptive_beta = 0.8 + 1.7 * uncertainty_ratio
    else:
        adaptive_beta = np.full_like(sigma_safe, beta)
    
    ucb = f_best - mu + adaptive_beta * sigma_safe
    score = ucb / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score


def _cost_aware_ei(mu, sigma, f_best, cost, xi=0.0):
    """
    Cost-aware Expected Improvement with smooth cost normalization.
    """
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # EI for minimization
    imp = f_best - mu - xi
    z = imp / sigma_safe
    ei = imp * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Smooth cost normalization: use soft-min to avoid discontinuities
    # This creates a smoother preference for cheaper points
    log_cost = np.log(cost_safe)
    log_cost_min = np.min(log_cost)
    
    # Soft normalization: exp(-alpha * (log_cost - log_cost_min))
    alpha = 0.5  # Temperature for cost preference
    cost_weight = np.exp(-alpha * (log_cost - log_cost_min))
    
    score = ei * cost_weight
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score


def _hybrid_acquisition_v2(mu, sigma, f_best, cost):
    """
    Improved hybrid acquisition with smoother blending.
    """
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute both scores
    ucb_score = _cost_aware_ucb(mu, sigma, f_best, cost, beta=1.5)
    ei_score = _cost_aware_ei(mu, sigma, f_best, cost, xi=0.0)
    
    # Normalize scores to comparable scale
    ucb_max = np.max(ucb_score)
    ei_max = np.max(ei_score)
    
    if ucb_max > 1e-12:
        ucb_norm = ucb_score / ucb_max
    else:
        ucb_norm = np.zeros_like(ucb_score)
    
    if ei_max > 1e-12:
        ei_norm = ei_score / ei_max
    else:
        ei_norm = np.zeros_like(ei_score)
    
    # Adaptive weighting based on uncertainty and improvement potential
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-8:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Weight: use sigmoid-like transition for smooth blending
    # More EI when uncertainty is low (exploit), more UCB when high (explore)
    temperature = 0.3
    w_ucb = 1.0 / (1.0 + np.exp(-temperature * (sigma_norm - 0.5)))
    w_ei = 1.0 - w_ucb
    
    score = w_ucb * ucb_norm + w_ei * ei_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean predictions from the GP.
    sigma : ndarray, shape (2000,)
        Standard deviation predictions from the GP.
    f_best : float
        Best observed function value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    start_time = time.time()
    deadline = 280.0  # Safety margin before 300s limit
    
    # Validate inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    if mu.shape != (2000,) or sigma.shape != (2000,) or cost.shape != (2000,):
        raise ValueError("Input shapes must be (2000,)")
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        # Return uniform scores if inputs contain non-finite values
        return np.ones(2000, dtype=np.float64)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)
    
    # Use improved hybrid acquisition
    scores = _hybrid_acquisition_v2(mu, sigma, f_best, cost)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return current scores as safety measure
        pass
    
    # Final validation: ensure all scores are finite
    if not np.all(np.isfinite(scores)):
        # Replace non-finite values with a reasonable default
        finite_mask = np.isfinite(scores)
        if np.any(finite_mask):
            default_val = np.median(scores[finite_mask])
        else:
            default_val = 0.0
        scores = np.where(np.isfinite(scores), scores, default_val)
    
    return scores
# EVOLVE-BLOCK-END
