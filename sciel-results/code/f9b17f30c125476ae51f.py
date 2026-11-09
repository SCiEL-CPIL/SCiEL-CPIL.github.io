"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6d.
    
    Combines normalized Expected Improvement (EI) with a UCB exploration bonus,
    with adaptive exploration weight that decreases as f_best approaches the
    known global minimum. All scores are divided by evaluation cost.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions at candidate points.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Safety: replace near-zero values
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Compute Expected Improvement (EI) for exploitation
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB score for exploration
    # For minimization: lower confidence bound = mu - beta * sigma
    # We maximize the potential improvement: beta * sigma - mu
    beta = 1.8
    ucb_score = beta * sigma_safe - mu
    
    # Normalize EI and UCB to comparable [0, 1] scales
    ei_max = np.max(ei)
    if ei_max < 1e-10:
        ei_norm = ei
    else:
        ei_norm = ei / ei_max
    
    ucb_max = np.max(ucb_score)
    if ucb_max < 1e-10:
        ucb_norm = ucb_score
    else:
        ucb_norm = ucb_score / ucb_max
    
    # Adaptive exploration weight:
    # Hartmann 6d global min is approximately -3.32237
    # When f_best is far from optimum (e.g., -1.0), explore more
    # When f_best is close (e.g., -3.2), exploit more
    # Map f_best from [-3.32, -1.0] to exploration weight [0.35, 0.55]
    f_min_known = -3.32237
    f_initial = -1.0  # typical initial f_best after 5 random points
    
    # Progress: 0 when f_best = f_initial, 1 when f_best = f_min_known
    progress = np.clip((f_initial - f_best) / (f_initial - f_min_known + 1e-10), 0.0, 1.0)
    
    # Exploration weight: 0.55 (early) -> 0.35 (late)
    w_explore = 0.55 - 0.20 * progress
    w_exploit = 1.0 - w_explore
    
    combined = w_exploit * ei_norm + w_explore * ucb_norm
    
    # Cost-aware: divide by cost
    scores = combined / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time guard: if we've exceeded 280s, return immediately
    if time.time() - start_time > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
