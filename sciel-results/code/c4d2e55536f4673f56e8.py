"""Rewritten program with a stronger cost-aware acquisition for rastrigin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _expected_improvement(mu, sigma, f_best):
    """Standard EI with a small floor on sigma to avoid division by zero."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return ei


def _ucb(mu, sigma, f_best, beta):
    """Upper confidence bound (maximization form)."""
    return mu + beta * sigma


def _pi(mu, sigma, f_best):
    """Probability of improvement."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    return norm.cdf(z)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing rastrigin 2d.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute EI
    ei = _expected_improvement(mu, sigma, f_best)
    
    # Check time
    if time.time() > deadline:
        # Fallback to simple EI/cost
        scores = ei / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Adaptive beta based on how close we are to the optimum
    # When f_best is small (close to 0), we want more exploitation (lower beta)
    # When f_best is large, we want more exploration (higher beta)
    # f_best for rastrigin ranges from 0 to ~60
    f_best_clamped = max(f_best, 1e-6)
    # Beta decreases as f_best approaches 0
    beta = 2.0 * np.exp(-0.1 * f_best_clamped) + 0.5
    
    ucb = _ucb(mu, sigma, f_best, beta)
    
    # Check time
    if time.time() > deadline:
        scores = (ei + ucb) / (2.0 * cost_safe)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Use a weighted combination that adapts to the current state
    # Normalize EI and UCB by their respective scales
    ei_scale = np.std(ei) + 1e-9
    ucb_scale = np.std(ucb) + 1e-9
    
    ei_norm = ei / ei_scale
    ucb_norm = ucb / ucb_scale
    
    # Weight: more weight to EI when f_best is small (exploitation phase)
    # More weight to UCB when f_best is large (exploration phase)
    ei_weight = 0.5 + 0.3 * np.exp(-0.1 * f_best_clamped)
    ucb_weight = 1.0 - ei_weight
    
    combined = ei_weight * ei_norm + ucb_weight * ucb_norm
    
    # Cost-aware normalization: divide by cost to prefer cheaper evaluations
    # Use a power less than 1 to soften the cost penalty
    scores = combined / (cost_safe ** 0.5)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
