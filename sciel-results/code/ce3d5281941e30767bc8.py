"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Uses Probability of Improvement (PI) for a structurally different
    exploitation-exploration balance compared to EI or UCB.
    PI focuses on the likelihood of finding a point better than f_best,
    rather than the expected magnitude of improvement. This can be
    beneficial for multimodal functions like Branin by preventing
    excessive exploitation of a single local minimum.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute z-score for Probability of Improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement for minimization
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    pi_scores = norm.cdf(z)
    
    # Add a small sigma-weighted exploration term to differentiate
    # candidates with similar PI values. This helps explore uncertain
    # regions without overwhelming the exploitation signal.
    # The scale is small (0.01) so PI dominates.
    exploration = 0.01 * sigma_safe / (np.max(sigma_safe) + eps)
    
    scores = pi_scores + exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
