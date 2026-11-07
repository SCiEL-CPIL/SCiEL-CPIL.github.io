import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining EI structure with adaptive exploration.
    Uses Expected Improvement (EI) as the base, which is known to be strong
    for small-budget BO. We add a small, adaptive exploration boost based
    on the ratio of sigma to the current improvement scale to avoid getting
    stuck in local optima too early, but keep it conservative.
    
    The score is designed to be maximized.
    """
    # Deadline guard: track elapsed time to ensure we return before timeout
    start_time = time.time()
    
    # Standard EI calculation
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Handle sigma near zero to avoid division by zero
    eps = 1e-9
    sigma_safe = np.where(sigma < eps, eps, sigma)
    
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Check time budget
    if time.time() - start_time > 250:
        # If we're running out of time, just return EI
        scores = ei
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        return scores
    
    # Adaptive exploration: add a small bonus for high uncertainty
    # But scale it down as we get more confident (as f_best improves or sigma decreases)
    # We want to be slightly more exploratory than pure EI in early stages
    
    # Scale exploration by the relative uncertainty
    # If sigma is large relative to the typical improvement, explore more
    # Use a soft cap to prevent excessive exploration
    sigma_mean = np.mean(sigma_safe)
    if sigma_mean > 0:
        # Exploration weight: starts at ~0.1 and decays as sigma_mean decreases
        # This is a heuristic: as the GP becomes more confident, rely more on exploitation
        exploration_weight = 0.1 * (sigma_mean / (np.std(mu) + sigma_mean + eps))
    else:
        exploration_weight = 0.0
    
    # Add exploration bonus: weighted by sigma, but normalized
    # This encourages sampling from uncertain regions without dominating the EI signal
    exploration_bonus = exploration_weight * sigma_safe
    
    scores = ei + exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
