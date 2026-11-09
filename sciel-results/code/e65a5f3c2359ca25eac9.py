"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard (330s wall-clock limit, use 300s for safety)
_DEADLINE = None
_DEADLINE_MARGIN = 5.0  # seconds before the hard limit

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 300.0
    return _DEADLINE

def _time_remaining():
    return max(0.0, _get_deadline() - time.time() - _DEADLINE_MARGIN)


def _expected_improvement(mu, sigma, f_best):
    """Standard expected improvement."""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)


def _ucb(mu, sigma, f_best, xi=0.0, kappa=2.0):
    """Upper Confidence Bound for minimization."""
    return mu - kappa * sigma + xi


def _pi(mu, sigma, f_best, xi=0.0):
    """Probability of improvement."""
    sigma_safe = np.maximum(sigma, 1e-12)
    return norm.cdf((f_best - mu - xi) / sigma_safe)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Styblinski-Tang 2D.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each point (1 + mean(x_scaled))
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure deadline is set
    _get_deadline()
    
    # Safety: if time is running out, return simple EI/cost
    if _time_remaining() < 0.5:
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei / np.maximum(cost, 1e-9)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute multiple acquisition functions and combine them
    # 1. Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. UCB (for minimization, lower is better, so we want high values of -mu + kappa*sigma)
    kappa = 2.5
    ucb = -mu + kappa * sigma_safe
    
    # 3. Probability of Improvement
    pi = norm.cdf((f_best - mu) / sigma_safe)
    
    # 4. Expected Improvement per unit cost
    ei_cost = ei / cost_safe
    
    # 5. UCB per unit cost
    ucb_cost = ucb / cost_safe
    
    # 6. PI per unit cost
    pi_cost = pi / cost_safe
    
    # Combine: weight EI/cost heavily, with some UCB/cost and PI/cost
    # The cost-aware aspect is crucial since evaluating expensive points should be penalized
    # Use a blend that emphasizes cost-effectiveness
    scores = 0.6 * ei_cost + 0.3 * ucb_cost + 0.1 * pi_cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
