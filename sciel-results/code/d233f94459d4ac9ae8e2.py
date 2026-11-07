"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline management
_START_TIME = time.time()
_DEADLINE_SECONDS = 285.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        return np.zeros_like(mu, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle any NaN or Inf values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure sigma is strictly positive for division
    sigma = np.maximum(sigma, 1e-12)
    
    # Ensure cost is within reasonable bounds for Rastrigin task
    # Cost is 1 + mean(x_scaled), so [1, 2]
    cost = np.clip(cost, 1.0, 2.0)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB (Upper Confidence Bound) for minimization
    # We use a dynamic kappa based on the uncertainty level
    # Higher average sigma -> higher kappa (more exploration)
    # Lower average sigma -> lower kappa (more exploitation)
    mean_sigma = np.mean(sigma)
    # Normalize mean_sigma to [0, 1] roughly. 
    # If mean_sigma is very small, we are confident, explore less.
    # If mean_sigma is large, we are uncertain, explore more.
    # Rastrigin values can be large, but sigma is in objective units.
    # A reasonable range for sigma in this context might be 0.1 to 10.
    # We map mean_sigma to a kappa between 1.0 and 3.0
    kappa = 1.0 + 2.0 * np.tanh(mean_sigma)
    
    ucb = mu - kappa * sigma
    
    # Normalize EI and UCB to [0, 1]
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    if ucb_max - ucb_min > 1e-12:
        ucb_norm = 1.0 - (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Adaptive Weighting:
    # If the model is very uncertain (high sigma), trust UCB more.
    # If the model is certain (low sigma), trust EI more.
    # weight_ucb is proportional to the relative uncertainty.
    # We use the ratio of mean_sigma to the scale of the objective (approx f_best or 1.0)
    # Let's use a simple sigmoid on the mean_sigma to get a weight in [0.2, 0.8]
    # If mean_sigma is high, weight_ucb -> 0.8. If low, weight_ucb -> 0.2.
    weight_ucb = 0.2 + 0.6 * np.tanh(mean_sigma * 2.0)
    weight_ei = 1.0 - weight_ucb
    
    combined = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Cost-aware scoring:
    # Use a power-law cost penalty to strongly favor low-cost candidates
    # cost is in [1, 2]. cost^1.5 amplifies the difference between 1.0 and 2.0
    # 1.0^1.5 = 1.0, 2.0^1.5 = 2.828. 
    # This is a stronger penalty than linear cost.
    cost_penalty = cost ** 1.5
    
    score = combined / cost_penalty
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
