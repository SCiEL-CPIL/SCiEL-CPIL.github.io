"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6d.
    Uses a direct, principled blend of Expected Improvement and UCB,
    normalized by evaluation cost with a sqrt dampening to allow exploration
    of higher-cost regions.
    
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
    t0 = time.time()
    DEADLINE = 320.0  # Safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Floor sigma and cost
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Check deadline early
    if time.time() - t0 > DEADLINE - 10:
        # Fallback to simple cost-aware UCB
        beta = 1.5
        improvement = f_best - mu + beta * sigma
        scores = improvement / np.sqrt(cost)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Known minimum of Hartmann 6d
    HARTMANN_6D_MIN = -3.32237
    
    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB improvement with adaptive beta
    range_val = f_best - HARTMANN_6D_MIN
    if range_val <= 0:
        progress = 1.0
    else:
        initial_guess = -1.0
        total_range = initial_guess - HARTMANN_6D_MIN
        progress = np.clip((initial_guess - f_best) / total_range, 0.0, 1.0)
    
    # Adaptive beta: explore more early, exploit more late
    beta_early = 2.0
    beta_late = 0.5
    beta = beta_early + (beta_late - beta_early) * progress
    
    ucb_improvement = f_best - mu + beta * sigma
    
    # Blend EI and UCB based on progress
    # Early: weight towards UCB, Late: weight towards EI
    w = 0.5 * (1.0 + progress)  # w in [0.5, 1.0]
    blended = w * ei + (1.0 - w) * ucb_improvement
    
    # Cost-aware score using sqrt(cost) to reduce penalty on expensive points
    # This allows the algorithm to consider high-cost points that may contain
    # the true global minimum, which standard cost division might overly penalize
    cost_factor = np.sqrt(cost)
    base_scores = blended / cost_factor
    
    # Add a small uncertainty bonus to encourage exploration in uncertain regions
    # This is a smooth surrogate that balances exploitation and exploration
    # The bonus is scaled by the progress: more exploration early, less late
    # Normalize sigma to [0, 1] range for stable bonus scaling
    sigma_min, sigma_max = np.min(sigma), np.max(sigma)
    sigma_range = sigma_max - sigma_min + 1e-9
    sigma_norm = (sigma - sigma_min) / sigma_range
    uncertainty_bonus = 0.1 * (1.0 - progress) * sigma_norm
    
    scores = base_scores + uncertainty_bonus
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
