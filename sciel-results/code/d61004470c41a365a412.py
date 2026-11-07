"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Powell 4D.
    Combines Expected Improvement with adaptive UCB exploration,
    normalized and cost-discounted for robustness.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive UCB exploration
    # Estimate relative uncertainty to scale exploration
    med_sigma = np.median(sigma_safe)
    std_mu = np.std(mu) + 1e-8
    # beta scales with relative uncertainty: high sigma relative to mu spread -> more exploration
    beta = np.clip(med_sigma / std_mu, 0.1, 2.0)
    
    # UCB for minimization: -mu + beta * sigma (higher = more uncertain/low)
    ucb = -mu + beta * sigma_safe
    
    # Normalize both components to be comparable
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-8
    ei_norm = (ei - ei_mean) / ei_std
    
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb) + 1e-8
    ucb_norm = (ucb - ucb_mean) / ucb_std
    
    # Weighted combination: 70% EI, 30% UCB
    # This balances exploitation (EI) with exploration (UCB)
    score = 0.7 * ei_norm + 0.3 * ucb_norm
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    # Use a mild power to amplify cost preference without over-penalizing
    cost_factor = np.power(cost_safe, 0.5)
    score = score / cost_factor
    
    # Ensure all outputs are finite and non-negative where appropriate
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
