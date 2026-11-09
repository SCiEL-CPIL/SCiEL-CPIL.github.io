# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior
    f_best : float
        Best observed function value
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Deadline guard: track elapsed time
    t_start = time.time()
    deadline = t_start + 280  # Safety margin before 300s limit
    
    # Add small epsilon to avoid division by zero
    eps = 1e-10
    sigma_safe = sigma + eps
    cost_safe = cost + eps
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB (Upper Confidence Bound) - for minimization, we want low mu + high sigma
    # UCB for minimization: mu - beta * sigma (lower is better)
    # But for acquisition (higher score = better), we use: -mu + beta * sigma
    # beta adapts based on f_best: larger beta when f_best is large (early), smaller when f_best is small (late)
    
    # Adaptive beta: starts high for exploration, decreases as f_best improves
    # f_best for rastrigin can be up to ~200, minimum is 0
    # Use a smooth transition
    f_best_clipped = np.clip(f_best, 0.0, 100.0)
    
    # beta decreases as f_best decreases
    # When f_best is large (e.g., 50+), beta ~ 3.0 (heavy exploration)
    # When f_best is small (e.g., < 1), beta ~ 0.5 (heavy exploitation)
    beta = 0.5 + 2.5 * np.exp(-f_best_clipped / 5.0)
    
    # UCB score for minimization (higher is better for acquisition)
    ucb = -mu + beta * sigma
    
    # Cost-aware versions
    scores_ei = ei / cost_safe
    scores_ucb = ucb / cost_safe
    
    # Adaptive blending: more UCB early (high f_best), more EI late (low f_best)
    # weight_ucb is high when f_best is large
    weight_ucb = 0.3 + 0.5 * np.exp(-f_best_clipped / 3.0)
    weight_ei = 1.0 - weight_ucb
    
    scores = weight_ei * scores_ei + weight_ucb * scores_ucb
    
    # Check time
    if time.time() > deadline:
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Add a small pure uncertainty bonus to help escape local minima
    # This is weighted to be small but nonzero
    scores_unc = sigma / cost_safe
    scores = 0.95 * scores + 0.05 * scores_unc
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
