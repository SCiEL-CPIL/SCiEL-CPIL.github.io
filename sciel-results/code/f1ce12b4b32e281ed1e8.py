"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track total elapsed time for deadline guard
_T0 = time.time()
_DEADLINE = 320.0

def _time_left():
    return _DEADLINE - (time.time() - _T0)

def acquisition(mu, sigma, f_best, cost):
    """
    Computes a cost-aware acquisition function combining Expected Improvement (EI)
    and Upper Confidence Bound (UCB) with adaptive exploration and diversity penalty.
    
    Args:
        mu: array of shape (N,) predicted mean
        sigma: array of shape (N,) predicted std deviation
        f_best: float, best value found so far
        cost: array of shape (N,) evaluation cost for each candidate
        
    Returns:
        scores: array of shape (N,) acquisition values (higher is better)
    """
    # Deadline guard: if we've been running too long, return a simple, fast score
    if _time_left() < 10.0:
        # Fallback: simple EI per cost, very fast
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return ei / cost_safe

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma and cost to small positive values
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate PDF and CDF
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    ei = np.maximum(ei, 0.0)
    
    # UCB with adaptive beta
    # For minimization, we want to maximize (beta * sigma - mu)
    # But we also want to exploit low mu, so we use a combination.
    # Let's use a standard UCB form for minimization: -mu + beta * sigma
    # However, EI already captures exploitation and exploration.
    # Let's add a UCB-like term to encourage exploration in high uncertainty regions.
    beta = 2.0
    ucb = -mu + beta * sigma_safe
    
    # Normalize EI and UCB to combine them fairly
    ei_max = np.max(ei) if np.max(ei) > 1e-12 else 1.0
    ucb_max = np.max(ucb) if np.max(ucb) > 1e-12 else 1.0
    ucb_min = np.min(ucb) if np.max(ucb) > 1e-12 else 0.0
    
    ei_norm = ei / ei_max
    ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min + 1e-12)
    
    # Weighted combination: 70% EI, 30% UCB
    combined = 0.7 * ei_norm + 0.3 * ucb_norm
    
    # Diversity penalty: penalize candidates in well-explored (low sigma) regions
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    if sigma_max - sigma_min > 1e-12:
        sigma_norm = (sigma_safe - sigma_min) / (sigma_max - sigma_min)
    else:
        sigma_norm = np.ones_like(sigma_safe)
    
    # Crowding is high when sigma_norm is low.
    crowding = 1.0 - sigma_norm
    
    # Penalty weight: how strongly to penalize crowded regions.
    penalty_weight = 0.3
    # Scale penalty by the magnitude of the combined score
    combined_max = np.max(combined) if np.max(combined) > 1e-12 else 1.0
    crowding_penalty = penalty_weight * crowding * combined_max
    
    # Combined score: combined - crowding_penalty
    score = combined - crowding_penalty
    
    # Cost-aware: divide by cost (higher cost = lower priority)
    score = score / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
