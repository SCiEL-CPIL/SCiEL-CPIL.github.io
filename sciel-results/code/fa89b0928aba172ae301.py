"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for the session
_SESSION_START = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed function value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate, shape (2000,).
        
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Safety deadline guard: if we've used > 295s of the 330s limit,
    # return a simple fallback immediately.
    elapsed = time.time() - _SESSION_START
    if elapsed > 295:
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        z_clipped = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
        score = ei / np.maximum(cost, 1e-8)
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure sigma and cost are positive
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Lower Confidence Bound (for minimization): LCB = mu - kappa * sigma
    # Score component: -LCB = -mu + kappa * sigma
    # Use a higher kappa for better exploration in multimodal landscapes
    kappa = 2.0
    lcb_score = -mu + kappa * sigma_safe
    
    # Cost-normalized versions
    ei_cost = ei / cost_safe
    lcb_cost = lcb_score / cost_safe
    
    # Adaptive weighting strategy:
    # We want to explore more (LCB) when f_best is high (early in optimization)
    # and exploit more (EI) when f_best is low (late in optimization).
    
    # Rastrigin 2D on [-5.12, 5.12]^2 has values typically in [0, 40].
    # f_best starts high and decreases.
    
    # Sigmoid transition centered around f_best = 10.
    # When f_best > 10, alpha (weight for EI) should be low.
    # When f_best < 10, alpha (weight for EI) should be high.
    
    # alpha = 1 / (1 + exp(k * (f_best - center)))
    # If f_best is large, exp is large, alpha -> 0 (more LCB).
    # If f_best is small, exp is small, alpha -> 1 (more EI).
    
    k = 5.0
    center = 10.0
    
    # Clip f_best to avoid overflow in exp
    f_best_clipped = np.clip(f_best, 0, 50)
    
    alpha = 1.0 / (1.0 + np.exp(k * (f_best_clipped - center)))
    
    # Ensure alpha is within [0, 1]
    alpha = np.clip(alpha, 0.0, 1.0)
    
    # Combine EI and LCB
    score = alpha * ei_cost + (1 - alpha) * lcb_cost
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
