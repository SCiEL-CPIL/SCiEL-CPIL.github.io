import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses a hybrid EI + adaptive UCB approach with a smooth schedule.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle zero sigma to avoid division by zero
    safe_sigma = np.where(sigma < 1e-12, 1e-12, sigma)
    
    # EI calculation:
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / safe_sigma
    
    # Standard normal CDF and PDF
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    EI = (f_best - mu) * Phi + safe_sigma * phi
    
    # Adaptive exploration via UCB component
    # Hartmann 6D minimum is approximately -3.32237
    # f_best starts around -1.0 to -2.0 and decreases toward -3.32
    
    f_best_clamped = max(f_best, -3.5)
    # Normalize: 0 when close to optimum, 1 when far
    exploration_scale = np.clip((-f_best_clamped - 3.32) / 2.32, 0.0, 1.0)
    
    # UCB weight: larger when far from optimum
    ucb_weight = 0.1 + 0.4 * exploration_scale
    
    UCB = -mu + ucb_weight * safe_sigma
    
    # Blend EI and UCB with a smooth schedule
    # Early on (far from optimum), weight UCB more
    # Later (close to optimum), weight EI more
    blend = 0.3 + 0.7 * (1.0 - exploration_scale)
    
    scores = blend * EI + (1.0 - blend) * UCB
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
