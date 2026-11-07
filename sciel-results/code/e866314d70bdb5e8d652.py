"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Uses Variance-Gated Expected Improvement (VEI) combined with a UCB term.
    
    This structural change addresses the multimodality of Branin-2D by
    explicitly gating the exploitation signal (EI) with the relative
    uncertainty (sigma). This forces the search to prioritize regions where
    the GP is uncertain, helping to escape local minima. A baseline UCB term
    is added to ensure continued exploration even where predicted improvement
    is low.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points (objective units).
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points (objective units).
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Ensure inputs are numpy arrays for efficient vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety margin: sigma must be positive for division and log/exp operations
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Compute Standard Expected Improvement (EI)
    # Improvement is (f_best - mu) for minimization
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = improvement * Phi_z + sigma_safe * phi_z
    
    # 2. Compute Variance Gate
    # Scale EI by (sigma / sigma_max)^gamma
    # This suppresses EI in well-explored (low sigma) regions
    # and amplifies it in unexplored (high sigma) regions.
    sigma_max = np.max(sigma_safe)
    if sigma_max > eps:
        gamma = 1.5
        # Clip ratio to [0, 1] to avoid numerical issues if sigma_max is tiny
        ratio = np.clip(sigma_safe / sigma_max, 0.0, 1.0)
        variance_gate = ratio ** gamma
    else:
        variance_gate = np.ones_like(sigma_safe)
    
    ei_gated = ei * variance_gate
    
    # 3. Combine with UCB term
    # The inspiration showed that a UCB with beta=1.0 is very effective (score 1.42).
    # We add a UCB term to the gated EI. This ensures that even if EI is 0
    # (e.g., mu > f_best), we still get a score proportional to sigma,
    # driving exploration.
    beta = 1.0
    ucb_term = beta * sigma_safe
    
    scores = ei_gated + ucb_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
