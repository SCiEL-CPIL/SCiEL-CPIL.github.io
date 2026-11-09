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
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate Expected Improvement (for minimization)
    # EI = E[f_best - f | f < f_best] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Lower Confidence Bound (for minimization): LCB = mu - kappa * sigma
    # We want to maximize the "improvement" over LCB, or equivalently minimize LCB
    # The acquisition should favor points with low LCB (high improvement potential)
    # So we use -LCB = -mu + kappa * sigma as a score component
    kappa = 1.0
    lcb_score = -mu + kappa * sigma_safe
    
    # Also consider a PI (Probability of Improvement) based component
    pi = norm.cdf(z_clipped)
    
    # Cost-normalized versions
    ei_cost = ei / cost_safe
    lcb_cost = lcb_score / cost_safe
    pi_cost = pi / cost_safe
    
    # Adaptive weighting strategy:
    # When f_best is still high (early in optimization), explore more (LCB/PI weight)
    # When f_best is close to optimum, exploit more (EI weight)
    # Use a smooth transition based on f_best relative to typical rastrigin values
    
    # Rastrigin minimum is 0, typical values in [-5.12, 5.12]^2 are 0-40+
    # f_best starts high and decreases
    # Use a sigmoid to blend: high f_best -> more exploration, low f_best -> more exploitation
    
    # Normalize f_best to [0, 1] range approximately (0 to 40 covers most of the range)
    f_best_norm = np.clip(f_best / 40.0, 0.0, 1.0)
    
    # Weight for exploration increases when f_best is high
    # alpha_explore: weight for LCB, alpha_ei: weight for EI
    # When f_best_norm is high (early), alpha_explore is high
    # When f_best_norm is low (late), alpha_ei is high
    
    # Use a smooth blend:
    # alpha_explore = 0.5 * (1 + f_best_norm)  # ranges from 0.5 to 1.0
    # alpha_ei = 0.5 * (1 - f_best_norm)     # ranges from 0.5 to 0.0
    # But we also want PI to contribute
    
    # Let's use a three-way blend:
    # w_ei = 0.4 + 0.4 * (1 - f_best_norm)   # 0.4 to 0.8
    # w_lcb = 0.3 + 0.3 * f_best_norm        # 0.3 to 0.6
    # w_pi = 0.3 - 0.3 * f_best_norm         # 0.3 to 0.0 (but keep min 0.1)
    
    w_ei = 0.4 + 0.4 * (1.0 - f_best_norm)
    w_lcb = 0.3 + 0.3 * f_best_norm
    w_pi = max(0.1, 0.3 - 0.3 * f_best_norm)
    
    # Normalize weights to sum to 1
    w_sum = w_ei + w_lcb + w_pi
    w_ei /= w_sum
    w_lcb /= w_sum
    w_pi /= w_sum
    
    # Combine scores
    score = w_ei * ei_cost + w_lcb * lcb_cost + w_pi * pi_cost
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
