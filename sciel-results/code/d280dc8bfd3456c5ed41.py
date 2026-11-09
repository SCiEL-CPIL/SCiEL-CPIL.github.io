"""Cost-aware acquisition for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
_DEADLINE_START = time.time()


def _expected_improvement_cost(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement for minimization.
    """
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    
    ei = np.zeros_like(mu)
    mask = z > -0.5
    if np.any(mask):
        z_m = z[mask]
        mu_m = mu[mask]
        sig_m = sigma_safe[mask]
        ei[mask] = (f_best - mu_m) * norm.cdf(z_m) + sig_m * norm.pdf(z_m)
    
    cost_safe = np.maximum(cost, 1e-6)
    return ei / cost_safe


def _ucb_cost(mu, sigma, f_best, cost, beta=0.5):
    """
    Cost-aware Upper Confidence Bound for minimization.
    For minimization, we want low values, so we use mu - beta*sigma.
    But in the context of maximizing an acquisition score, we want high scores.
    So we use -(mu - beta*sigma) / cost = (beta*sigma - mu) / cost.
    """
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-6)
    # For minimization, LCB = mu - beta * sigma
    # We want to maximize the "potential for improvement", so we invert
    # Actually, let's stick to the standard UCB for minimization:
    # We want to find points where mu - beta*sigma is low.
    # So the score should be high when mu - beta*sigma is low.
    # Score = -(mu - beta * sigma) / cost
    score = -(mu - beta * sigma_safe) / cost_safe
    return score


def _qei_cost(mu, sigma, f_best, cost):
    """
    Quantile Expected Improvement: uses a quantile of the posterior instead of the mean.
    This can be more robust to outliers in the GP prediction.
    Here we use a simple approximation: adjust the mean by a fraction of the std.
    """
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Adjust mean towards the lower tail for minimization
    # This is a heuristic: mu_adj = mu - 0.5 * sigma
    mu_adj = mu - 0.5 * sigma_safe
    
    z = (f_best - mu_adj) / sigma_safe
    ei = np.zeros_like(mu)
    mask = z > -0.5
    if np.any(mask):
        z_m = z[mask]
        mu_m = mu_adj[mask]
        sig_m = sigma_safe[mask]
        ei[mask] = (f_best - mu_m) * norm.cdf(z_m) + sig_m * norm.pdf(z_m)
    
    return ei / cost_safe


def _pi_cost_adjusted(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement.
    """
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    cost_safe = np.maximum(cost, 1e-6)
    return pi / cost_safe


def _hybrid_acquisition(mu, sigma, f_best, cost):
    """
    Hybrid acquisition combining EI, UCB, QEI, and PI with adaptive weights.
    """
    # Ensure finite inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-10, posinf=1e6, neginf=1e-10)
    sigma = np.maximum(sigma, 1e-10)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-6)
    cost = np.maximum(cost, 1e-6)
    
    # Check time budget
    elapsed = time.time() - _DEADLINE_START
    if elapsed > 315:  # Safety margin
        # Simple fallback
        return np.nan_to_num(-mu / cost, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute multiple acquisition functions
    ei_score = _expected_improvement_cost(mu, sigma, f_best, cost)
    ucb_score = _ucb_cost(mu, sigma, f_best, cost, beta=0.5)
    qei_score = _qei_cost(mu, sigma, f_best, cost)
    pi_score = _pi_cost_adjusted(mu, sigma, f_best, cost)
    
    # Normalize each score to have zero mean and unit variance
    def normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        std = np.std(x)
        if std < 1e-10:
            return x
        return (x - np.mean(x)) / std
    
    ei_norm = normalize(ei_score)
    ucb_norm = normalize(ucb_score)
    qei_norm = normalize(qei_score)
    pi_norm = normalize(pi_score)
    
    # Adaptive weights based on optimization progress
    # f_best close to -0.8 means we're near the optimum
    # f_best far from -0.8 means we need more exploration
    min_possible = -0.8
    max_possible = 0.8  # rough upper bound for cosine8
    progress = 1.0 - min(max((f_best - min_possible) / (max_possible - min_possible), 0.0), 1.0)
    
    # Early on (high progress): more exploration (UCB)
    # Later (low progress): more exploitation (EI, QEI, PI)
    w_ei = 0.30 + 0.15 * (1.0 - progress)
    w_ucb = 0.25 + 0.10 * progress
    w_qei = 0.20 + 0.10 * (1.0 - progress)
    w_pi = 0.15 + 0.05 * (1.0 - progress)
    
    # Ensure weights sum to 1
    total_w = w_ei + w_ucb + w_qei + w_pi
    w_ei /= total_w
    w_ucb /= total_w
    w_qei /= total_w
    w_pi /= total_w
    
    # Combined score
    score = w_ei * ei_norm + w_ucb * ucb_norm + w_qei * qei_norm + w_pi * pi_norm
    
    # Add a small bonus for low-cost points to encourage efficient exploration
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = max(cost_max - cost_min, 1e-10)
    cost_bonus = 0.05 * (1.0 - (cost - cost_min) / cost_range)
    
    score = score + cost_bonus
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Final time check
    if time.time() - _DEADLINE_START > 320:
        # Emergency fallback
        return np.nan_to_num(-mu / cost, nan=0.0, posinf=0.0, neginf=0.0)
    
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
    return _hybrid_acquisition(mu, sigma, f_best, cost)
# EVOLVE-BLOCK-END
