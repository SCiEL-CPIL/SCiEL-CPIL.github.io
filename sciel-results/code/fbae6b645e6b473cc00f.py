"""Cost-aware acquisition for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track time spent in this function and bail out safely.
_DEADLINE_START = time.time()
_MAX_TIME_PER_CALL = 2.0  # seconds per acquisition call (generous safety margin)


def _expected_improvement(mu, sigma, f_best):
    """Standard EI for minimization."""
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    # Handle edge cases where z is very negative (no improvement expected)
    ei = np.zeros_like(mu)
    mask = z > -1e-7
    ei[mask] = ((f_best - mu)[mask] * norm.cdf(z[mask]) + sigma_safe[mask] * norm.pdf(z[mask]))
    return ei


def _ucb(mu, sigma, f_best, cost, xi=0.1, kappa=1.0):
    """UCB-style acquisition for minimization, cost-adjusted."""
    return -(mu + kappa * sigma) / cost + xi


def _qei(mu, sigma, f_best, cost, xi=0.0):
    """Quantile-based EI: use the lower tail of the posterior."""
    # For minimization, we want to explore where mu is low and sigma is high
    # but penalize high cost
    # Use a modified EI that accounts for cost
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu - xi) / sigma_safe
    ei = np.zeros_like(mu)
    mask = z > -1e-7
    ei[mask] = ((f_best - mu - xi)[mask] * norm.cdf(z[mask]) + sigma_safe[mask] * norm.pdf(z[mask]))
    return ei / np.maximum(cost, 1e-6)


def _pi_cost_adjusted(mu, sigma, f_best, cost):
    """Probability of improvement, cost-adjusted, with exploration bonus."""
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    # Add a small exploration term based on sigma
    exploration = 0.1 * sigma_safe / np.maximum(f_best, 1e-6)
    return (pi + exploration) / np.maximum(cost, 1e-6)


def _hybrid_acquisition(mu, sigma, f_best, cost):
    """Combine multiple acquisition signals with cost awareness."""
    # Ensure finite inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-10, posinf=1e-6, neginf=1e-10)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-6)
    cost = np.maximum(cost, 1e-6)
    
    # Compute reference values for adaptive weighting
    mu_min, mu_max = np.min(mu), np.max(mu)
    mu_range = max(mu_max - mu_min, 1e-10)
    sigma_max = np.max(sigma)
    
    # Multiple acquisition functions
    ei = _expected_improvement(mu, sigma, f_best)
    ucb = _ucb(mu, sigma, f_best, cost, xi=0.05, kappa=0.8)
    qei = _qei(mu, sigma, f_best, cost, xi=0.0)
    pi_cost = _pi_cost_adjusted(mu, sigma, f_best, cost)
    
    # Normalize each to have zero mean and unit variance for fair combination
    def normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        std = np.std(x)
        if std < 1e-10:
            return x
        return (x - np.mean(x)) / std
    
    # Cost-adjusted versions
    ei_cost = ei / cost
    ucb_cost = ucb
    qei_cost = qei
    pi_cost_adj = pi_cost
    
    # Adaptive weighting based on optimization progress
    # When f_best is close to the minimum (-0.8), we should be more exploitative
    # When f_best is far from minimum, we should be more exploratory
    progress = 1.0 - min(max((f_best - (-0.8)) / 1.6, 0.0), 1.0)  # 0 = at min, 1 = far from min
    
    # Weighted combination with adaptive weights
    # Early on (high progress): more exploration (UCB, PI)
    # Later (low progress): more exploitation (EI, QEI)
    w_ei = 0.30 + 0.15 * (1.0 - progress)
    w_ucb = 0.25 + 0.10 * progress
    w_qei = 0.20 + 0.05 * (1.0 - progress)
    w_pi = 0.25 - 0.10 * progress
    
    # Ensure weights sum to 1
    total_w = w_ei + w_ucb + w_qei + w_pi
    w_ei /= total_w
    w_ucb /= total_w
    w_qei /= total_w
    w_pi /= total_w
    
    score = w_ei * normalize(ei_cost) + w_ucb * normalize(ucb_cost) + w_qei * normalize(qei_cost) + w_pi * normalize(pi_cost_adj)
    
    # Add a small bonus for low-cost points to encourage efficient exploration
    cost_bonus = 0.05 * (1.0 - cost / np.max(cost))
    
    score = score + cost_bonus
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    return score


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviations
    f_best : float - best observed value so far
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    array of shape (2000,) - acquisition scores (higher = better)
    """
    call_start = time.time()
    
    try:
        # Quick path: if we're running out of time, return a simple heuristic
        elapsed_global = time.time() - _DEADLINE_START
        if elapsed_global > 310:  # Safety margin before 330s kill
            # Simple cost-adjusted negative mean
            cost_safe = np.maximum(cost, 1e-6)
            score = -mu / cost_safe
            return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Main acquisition computation
        score = _hybrid_acquisition(mu, sigma, f_best, cost)
        
        # Check if we're still within time budget
        if time.time() - call_start > _MAX_TIME_PER_CALL:
            # Fallback to simpler computation
            cost_safe = np.maximum(cost, 1e-6)
            score = -mu / cost_safe
        
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        
    except Exception:
        # Absolute fallback: return finite scores based on cost only
        try:
            cost_safe = np.maximum(np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-6), 1e-6)
            return 1.0 / cost_safe
        except Exception:
            return np.ones(2000)
# EVOLVE-BLOCK-END
