# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 290.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Ackley 2D.
    
    Uses a hybrid of Probability of Improvement (PI) and Expected Improvement (EI)
    with adaptive blending based on f_best, normalized by sqrt(cost).
    
    The PI component provides a smoother, more stable gradient for exploration
    in multimodal landscapes, while EI drives exploitation. The blend weight
    shifts toward PI as f_best approaches 0 (optimum found), ensuring stability
    in late-stage optimization.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback: return simple EI-based scores to finish quickly
        sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-9)
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-9)
        mu_arr = np.asarray(mu, dtype=np.float64)
        z = (f_best - mu_arr) / sigma_safe
        ei = (f_best - mu_arr) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.maximum(ei, 0.0) / np.sqrt(cost_safe)
    
    # Convert to numpy arrays for safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score for improvement
    # Higher z means more likely to improve
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    pi = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to be non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive blending weight based on f_best
    # When f_best is close to 0 (good), weight more on PI for stability
    # When f_best is large (early), weight more on EI for exploitation
    # Ackley 2D on [-5,5]^2 has f_best range roughly [0, ~2]
    # Normalize f_best to [0, 1] for blending
    f_best_norm = np.clip(f_best / 2.0, 0.0, 1.0)
    
    # Weight for EI: higher when f_best is large (early stage)
    # Weight for PI: higher when f_best is small (late stage, near optimum)
    weight_ei = np.clip(f_best_norm, 0.2, 1.0)
    weight_pi = 1.0 - weight_ei
    
    # Scale EI to be comparable to PI (which is in [0, 1])
    # EI has units of objective value; scale by typical sigma magnitude
    # Use a robust scale estimate
    sigma_scale = np.median(sigma_safe) + 1e-9
    ei_scaled = ei / sigma_scale
    
    # Combine PI and EI
    combined = weight_pi * pi + weight_ei * ei_scaled
    
    # Cost-aware scoring: divide by sqrt(cost)
    # cost is in [1, 2] for this problem
    gamma = 0.5
    scores = combined / np.power(cost_safe, gamma)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
