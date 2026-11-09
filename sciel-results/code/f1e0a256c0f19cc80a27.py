"""Rewritten program using UCB acquisition with refined adaptive beta."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Uses Upper Confidence Bound (UCB) with refined adaptive beta parameter.
    """
    # Clip sigma to avoid division by zero and ensure stability
    sigma = np.maximum(sigma, 1e-9)
    
    # For minimization, we want to minimize mu - beta * sigma
    # UCB score = -mu + beta * sigma (higher is better)
    
    # Adaptive beta: start with higher exploration, decrease over time
    # Use f_best to estimate progress: as f_best approaches 0, reduce exploration
    # Normalize f_best to [0, 1] range for beta scheduling
    # f_best starts around 25 (5^2) and approaches 0
    f_norm = np.clip(f_best / 25.0, 0, 1)
    
    # Beta: use a smoother transition with slightly different range
    # Start at ~2.5 (high exploration), decrease to ~0.3 (strong exploitation)
    # Use quadratic decay for faster reduction in late stages
    beta = 0.3 + 2.2 * (f_norm ** 1.5)
    
    # UCB score: higher means more likely to be evaluated next
    # For minimization: we want low mu and high sigma (uncertainty)
    ucb_score = -mu + beta * sigma
    
    # Add a small penalty for extremely high sigma to prevent pure exploration
    # This keeps the search focused even in early stages
    sigma_penalty = 0.01 * np.maximum(sigma - 1.0, 0) ** 2
    ucb_score = ucb_score - sigma_penalty
    
    # Ensure all scores are finite
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, -np.inf)
    
    return ucb_score
# EVOLVE-BLOCK-END
