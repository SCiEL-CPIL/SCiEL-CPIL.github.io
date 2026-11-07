"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition with diversity penalty.
    
    Uses Expected Improvement as the primary exploitation signal, adds an
    adaptive UCB term for exploration, and applies a crowding penalty based
    on GP uncertainty (low sigma regions indicate well-sampled areas).
    Normalizes by evaluation cost to get improvement per unit cost.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value (lower is better).
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety margins
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -30.0, 30.0)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive UCB weight:
    # When uncertainty is high relative to the gap, explore more.
    sigma_mean = np.mean(sigma_safe)
    gap_scale = max(abs(f_best), 1e-10)
    
    # Adaptive beta: higher when uncertainty is significant relative to the gap
    uncertainty_ratio = min(sigma_mean / gap_scale, 5.0)
    beta = 0.1 + 0.9 * np.tanh(uncertainty_ratio)
    
    # Also consider the spread: if sigma varies a lot, trust high-sigma regions
    sigma_cv = np.std(sigma_safe) / (sigma_mean + 1e-12)
    beta = beta * (1.0 + 0.3 * min(sigma_cv, 2.0))
    
    # Cap beta to avoid over-exploration
    beta = min(beta, 1.5)
    
    ucb_term = beta * sigma_safe
    
    # Diversity penalty: penalize candidates in low-sigma (well-sampled) regions
    # Low sigma indicates the GP is confident, meaning the area is well-explored
    # We want to avoid redundant evaluations in dense clusters
    
    # Normalize sigma to [0, 1] for penalty calculation
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_range = max(sigma_max - sigma_min, 1e-12)
    sigma_norm = (sigma_safe - sigma_min) / sigma_range
    
    # Crowding penalty: high where sigma is low (well-sampled areas)
    # Use a soft penalty that's zero for high-sigma (uncertain) regions
    # and increases as sigma decreases
    crowding_weight = 0.3
    diversity_penalty = crowding_weight * (1.0 - sigma_norm) ** 2
    
    # Scale the penalty to be comparable to EI
    ei_scale = max(np.max(ei), 1e-10)
    diversity_penalty_scaled = diversity_penalty * ei_scale
    
    # Combined acquisition before cost normalization
    # Primary: EI (exploitation), Secondary: UCB (exploration), Tertiary: diversity
    combined = ei + ucb_term - diversity_penalty_scaled
    
    # Ensure combined doesn't go too negative (keep some exploration)
    combined = np.maximum(combined, 0.0)
    
    # Cost-aware: divide by cost to get improvement per unit cost
    score = combined / cost_safe
    
    # Ensure all values are finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    # Deadline guard
    elapsed = time.time() - t0
    if elapsed > 280.0:
        return score
    
    return score
# EVOLVE-BLOCK-END
