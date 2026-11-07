"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Combined GP-UCB and Expected Improvement acquisition function.
    
    Parameters
    ----------
    mu : array-like, shape (2000,)
        Posterior mean at candidate points.
    sigma : array-like, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values are preferred.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize EI to [0, 1] scale for combination
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # GP-UCB with time-varying beta
    # Use a reasonable estimate for the iteration count
    # Since we don't have direct access to t, we use a heuristic:
    # The number of evaluations affects the exploration level.
    # For the Beale function optimization with 30 total evaluations,
    # we can use a fixed but principled beta value.
    # beta_t = sqrt(2 * t * log(1 + t))
    # For t ~ 15 (mid-run), beta ~ sqrt(2*15*log(16)) ~ sqrt(30*2.77) ~ sqrt(83.1) ~ 9.1
    # This is quite large. Let's use a more moderate approach.
    
    # Alternative: use a fixed beta that balances exploration/exploitation
    # For 2D Beale function, moderate exploration is needed
    beta = 4.0  # Fixed exploration parameter
    
    # GP-UCB: mu + beta * sigma
    ucb = mu + beta * sigma
    
    # Normalize UCB for combination
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    if ucb_max - ucb_min > 1e-12:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Invert UCB norm since lower function values are better
    # We want to select points with low UCB (good mean) and high uncertainty
    # Actually, for minimization, we want low mu + beta*sigma
    # So the score should be negative of UCB, or we can use:
    # score = -mu + beta * sigma (exploit low mu, explore high sigma)
    
    # Better approach: use a weighted combination
    # Primary: exploit (low mu), Secondary: explore (high sigma)
    # score = -mu + alpha * sigma where alpha balances exploration
    
    # Let's use a simpler, more robust approach:
    # Combine normalized EI (exploitation) with normalized uncertainty (exploration)
    
    # Normalize sigma for exploration component
    sigma_max = np.max(sigma)
    if sigma_max > 1e-12:
        sigma_norm = sigma / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Weighted combination: 70% exploitation, 30% exploration
    weights_ei = 0.7
    weights_explore = 0.3
    
    scores = weights_ei * ei_norm + weights_explore * sigma_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
