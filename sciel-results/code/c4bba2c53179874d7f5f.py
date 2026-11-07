"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timing guard
_ACQ_START = time.time()
_TIME_LIMIT = 310.0  # Safety margin before 330s wall-clock limit


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    
    Uses Kriging Believer strategy: score is the negative GP mean at each candidate,
    but with a UCB-based gating mechanism to encourage exploration.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean at candidate points.
    sigma : array-like, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed objective value so far (lower is better).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more likely to be evaluated next.
    """
    # Check time budget
    elapsed = time.time() - _ACQ_START
    if elapsed > _TIME_LIMIT:
        # Return a safe fallback: pick the point with lowest mu
        mu_arr = np.asarray(mu)
        return -mu_arr

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Kriging Believer: score is negative of the GP mean
    # This favors points where the model predicts low values
    kb_scores = -mu
    
    # Add a small exploration bonus based on UCB to encourage visiting uncertain regions
    # UCB = mu - beta * sigma (for minimization, lower UCB is more promising)
    # We want to boost scores in regions with high uncertainty
    beta = 1.0
    ucb = mu - beta * sigma_safe
    
    # Normalize UCB to [0, 1] range for a smooth exploration bonus
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    if ucb_max - ucb_min > eps:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Exploration bonus: higher where UCB is low (more promising)
    # This creates a soft preference for exploring regions with both low predicted values and high uncertainty
    exploration_bonus = 0.1 * ucb_norm
    
    # Combine: Kriging Believer score + exploration bonus
    scores = kb_scores + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
