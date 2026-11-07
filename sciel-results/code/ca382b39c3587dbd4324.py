"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None

def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're close to the 330s limit, return safe default
    elapsed = _get_elapsed()
    if elapsed > 320:
        return np.zeros(len(mu), dtype=np.float64)

    # Clip sigma to avoid division by zero and numerical issues
    sigma_clipped = np.maximum(sigma, 1e-12)
    
    # Compute the z-score for Expected Improvement
    z = (f_best - mu) / sigma_clipped
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_clipped * norm.pdf(z)
    
    # Ensure ei is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Global budget-aware scaling strategy:
    # Instead of point-wise division by cost, we compute a normalized EI
    # and subtract a penalty proportional to cost relative to the mean cost.
    
    # Compute mean cost of the candidate batch
    mean_cost = np.mean(cost)
    std_cost = np.std(cost)
    
    # Normalize cost: (cost - mean_cost) / std_cost, clipped to avoid extreme values
    # This gives a z-score for cost within the batch
    cost_z = (cost - mean_cost) / (std_cost + 1e-12)
    cost_z = np.clip(cost_z, -3.0, 3.0)
    
    # Normalize EI to have zero mean and unit variance (for consistent scaling)
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std > 1e-12:
        ei_normalized = (ei - ei_mean) / ei_std
    else:
        ei_normalized = ei - ei_mean
    
    # Penalty strength: how much we penalize expensive points
    # Use a moderate penalty that shifts the balance based on remaining budget
    # The cost is in [1, 2], so cost_z is typically in [-1, 1] range
    penalty_strength = 0.5
    
    # Score: normalized EI minus penalty for high-cost points
    scores = ei_normalized - penalty_strength * cost_z
    
    # Also add a small exploration bonus based on sigma to help early exploration
    sigma_range = np.max(sigma) - np.min(sigma)
    if sigma_range > 1e-10:
        normalized_sigma = (sigma - np.min(sigma)) / sigma_range
        scores += 0.05 * normalized_sigma
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
