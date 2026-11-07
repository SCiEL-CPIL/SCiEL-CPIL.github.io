# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    Combines Expected Improvement (EI) with an adaptive UCB exploration bonus
    to balance exploration and exploitation across the optimization process.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety checks
    sigma = np.maximum(sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Compute Phi(z) and phi(z)
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Adaptive UCB exploration bonus
    # f_best starts high (up to 25) and decreases to 0.
    # We use a beta that starts higher (more exploration) and decreases (more exploitation).
    # This helps in early stages where EI can be too greedy.
    max_f = 25.0
    ratio = np.clip(f_best / max_f, 0.0, 1.0)
    
    # Beta schedule: starts at ~1.5 when f_best is high, decreases to ~0.5 when f_best is low
    # This provides strong exploration early, gentle exploration late
    beta = 0.5 + 1.0 * np.sqrt(ratio)
    
    # UCB component for minimization: -mu + beta * sigma
    # But we only want the exploration part (beta * sigma) as a bonus on top of EI
    # The -mu part is already captured in EI's exploitation component
    ucb_bonus = beta * sigma
    
    # Combine EI with adaptive UCB bonus
    # The UCB bonus is scaled down to not overwhelm EI
    score = ei + 0.3 * ucb_bonus
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
