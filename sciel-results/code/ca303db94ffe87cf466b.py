"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = None
_TIME_LIMIT = 320.0  # seconds, safety margin before 330s kill


def _ensure_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT
    return _DEADLINE


def _time_remaining():
    deadline = _ensure_deadline()
    return deadline - time.time()


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 3D.
    
    Uses a combination of Expected Improvement (EI) with a UCB-style exploration
    term to balance exploitation and exploration. The UCB weight adapts based on
    the spread of the posterior uncertainty.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more desirable to evaluate next.
    """
    global _DEADLINE
    _ensure_deadline()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    
    # Clamp sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB score: mu - beta * sigma (we want low mu, so we subtract)
    # For minimization, UCB = mu - beta * sigma
    # But we want higher score = more exploratory, so we use:
    # score = -mu + beta * sigma  (higher is better for exploration)
    
    # Adaptive beta: larger when we have less information (early on)
    # Use the fraction of points with high uncertainty as a proxy
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    # Beta that encourages exploration early, decreases as we learn
    # Scale beta based on the uncertainty relative to the improvement potential
    improvement_scale = max(f_best - np.min(mu), 1e-12)
    
    # Adaptive beta: use a fixed reasonable value, scaled by uncertainty
    # A common choice is beta = 2 * log(n * something) but we keep it simple
    # Use a moderate exploration constant
    beta = 1.5  # Moderate exploration
    
    # UCB score for minimization: we want to pick points with low mu and high sigma
    # ucb_score = -mu + beta * sigma (higher = more exploratory)
    ucb_score = -mu + beta * sigma_safe
    
    # Normalize both scores to [0, 1] range for combination
    ei_min, ei_max = np.min(ei), np.max(ei)
    ucb_min, ucb_max = np.min(ucb_score), np.max(ucb_score)
    
    if ei_max - ei_min > 1e-15:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.ones_like(ei)
    
    if ucb_max - ucb_min > 1e-15:
        ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.ones_like(ucb_score)
    
    # Combine: weight EI more heavily but keep exploration
    # The weight can adapt: more exploration when sigma is high overall
    exploration_weight = 0.4  # 40% UCB, 60% EI
    scores = (1 - exploration_weight) * ei_norm + exploration_weight * ucb_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
