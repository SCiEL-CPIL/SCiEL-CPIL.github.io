"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses a combination of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    to balance exploitation and exploration.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores. Higher values indicate more promising candidates.
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration: when f_best is still high, explore more;
    # when f_best is close to 0, exploit more.
    # f_best ranges from ~10 (Ackley max) down to 0 (minimum).
    # Map f_best to a progress indicator in [0, 1].
    progress = np.clip(1.0 - f_best / 10.0, 0.0, 1.0)
    
    # Interpolate beta: start at 1.5 (more exploration) -> end at 0.7 (more exploitation)
    beta = 1.5 - 0.8 * progress
    
    # Interpolate alpha: start at 0.75 (more UCB) -> end at 0.95 (more EI)
    alpha = 0.75 + 0.20 * progress
    
    # Upper Confidence Bound (UCB) for minimization
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and UCB with a weighted average
    scores = alpha * ei + (1 - alpha) * ucb
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
