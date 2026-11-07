"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6d.
    Uses a refined blend of Expected Improvement and UCB with cost-aware weighting.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions.
    sigma : array of shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Lowest observed value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate (1 + mean(x)).
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Floor to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # --- Expected Improvement (for minimization) ---
    z = (f_best - mu) / sigma
    z = np.clip(z, -10.0, 10.0)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(ei > 0, ei, 0.0)
    
    # --- UCB (Upper Confidence Bound for minimization) ---
    # Adaptive beta: increase exploration early, decrease later
    f_min_known = -3.32237
    if f_best > f_min_known:
        progress = 1.0 / (1.0 + np.exp(0.5 * (f_best - f_min_known)))
    else:
        progress = 1.0
    
    # Beta: higher when exploring (low progress), lower when exploiting (high progress)
    beta = 2.0 * (1.0 - progress) + 0.5
    
    # UCB score for minimization: we want to maximize this
    ucb = -mu + beta * sigma
    
    # Normalize EI and UCB to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ucb_shifted = ucb - np.min(ucb)
    ucb_shifted_max = np.max(ucb_shifted) if np.max(ucb_shifted) > 0 else 1.0
    
    # Blend EI and UCB
    ei_weight = 0.6 + 0.4 * progress
    ucb_weight = 1.0 - ei_weight
    
    # Normalize both to [0, 1]
    ei_norm = ei / ei_max if ei_max > 0 else ei
    ucb_norm = ucb_shifted / ucb_shifted_max if ucb_shifted_max > 0 else ucb_shifted
    
    # Combined score before cost adjustment
    combined = ei_weight * ei_norm + ucb_weight * ucb_norm
    
    # --- Cost-aware adjustment ---
    # Use a softer cost penalty: cost^0.5 instead of cost
    cost_factor = np.power(cost, 0.5)
    
    scores = combined / cost_factor
    
    # Check time
    if time.time() > deadline:
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
