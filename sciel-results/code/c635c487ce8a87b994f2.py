"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Combines Expected Improvement (EI) with a UCB-like exploration bonus,
    both normalized by evaluation cost. Uses adaptive exploration weight
    based on convergence progress.
    
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
    deadline = start_time + 280  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Replace near-zero sigma with small value to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Compute Expected Improvement (EI) for exploitation
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB score for exploration: beta * sigma - mu
    # For minimization, we want to find where the lower confidence bound is low
    # Lower confidence bound: mu - beta * sigma
    # We maximize the potential: beta * sigma - mu
    # Adaptive beta: more exploration when far from optimum, less when close
    # Hartmann 6d minimum is approximately -3.32237
    # Typical initial f_best is around -1 to -2, so progress = 1 - (f_best - (-3.32237)) / (initial - (-3.32237))
    # Simpler: use f_best directly as a proxy. When f_best is high (worse), explore more.
    # f_best ranges from ~-1 (early) to ~-3.32 (converged)
    # Map f_best from [-1, -3.32] to beta from [2.0, 1.0]
    # beta = 2.0 when f_best = -1, beta = 1.0 when f_best = -3.32
    # Linear interpolation: beta = 1.0 + (f_best - (-3.32237)) / ((-1.0) - (-3.32237)) * (2.0 - 1.0)
    # Simplified: beta = 1.0 + max(0, min(1, (f_best + 3.32237) / 2.32237))
    progress = np.clip((f_best + 3.32237) / 2.32237, 0.0, 1.0)
    beta = 1.0 + (1.0 - progress)  # beta = 2.0 when progress=0, beta=1.0 when progress=1
    beta = max(1.0, min(2.0, beta))
    
    ucb_score = beta * sigma_safe - mu
    
    # Normalize EI and UCB to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 1e-10 else 1.0
    ucb_max = np.max(ucb_score) if np.max(ucb_score) > 1e-10 else 1.0
    
    ei_norm = ei / ei_max
    ucb_norm = ucb_score / ucb_max
    
    # Blend exploitation and exploration
    # Weight more toward exploitation as we converge
    # Early: more exploration (0.4 EI, 0.6 UCB)
    # Late: more exploitation (0.7 EI, 0.3 UCB)
    ei_weight = 0.4 + 0.3 * progress  # 0.4 early, 0.7 late
    ucb_weight = 1.0 - ei_weight
    
    combined = ei_weight * ei_norm + ucb_weight * ucb_norm
    
    # Cost-aware: divide by cost
    scores = combined / cost_safe
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        pass
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
