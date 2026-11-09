"""Cost-aware acquisition function for Bayesian optimization on rastrigin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Set a deadline guard: we want to finish well within 300s
_DEADLINE = 280.0  # seconds
_start_time = time.time()


def _check_deadline():
    """Check if we're close to the deadline."""
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing rastrigin 2d.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-6)
    
    # Clamp sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Structural change: Adaptive cost-aware UCB with progress-based exploration decay.
    # 
    # Key insight: Rastrigin 2d is multimodal. Early in optimization, we need
    # strong exploration to discover different basins. As f_best approaches 0
    # (the known global minimum), we shift toward exploitation.
    #
    # We use a UCB-style acquisition: score = -mu + beta * sigma
    # where beta adapts based on progress (how close f_best is to 0).
    # The whole thing is normalized by cost to prefer cheaper evaluations.
    #
    # Progress metric: f_best starts high (rastrigin values can be ~20+) and
    # approaches 0. We map this to a beta that starts high and decays.
    
    # Compute progress: 1 when f_best is large (early), 0 when f_best ~ 0 (late)
    # Use a smooth transition. f_best for rastrigin 2d on [-5.12,5.12]^2 
    # starts around 10-20 and we want to reach ~0.
    # progress = 1 - min(f_best / f_max, 1.0) where f_max ~ 20
    f_max_estimate = 20.0
    progress = 1.0 - np.clip(f_best / f_max_estimate, 0.0, 1.0)
    # progress: 1.0 at start (f_best large), 0.0 when f_best ~ 0
    
    # Beta starts high for exploration, decays as we make progress
    # beta = beta_min + (beta_max - beta_min) * (1 - progress)
    # When progress=1 (early): beta = beta_max (high exploration)
    # When progress=0 (late): beta = beta_min (low exploration, exploitation)
    beta_max = 2.5
    beta_min = 0.3
    beta = beta_min + (beta_max - beta_min) * (1.0 - progress)
    
    # UCB score for minimization: we want to minimize, so lower mu is better.
    # score_ucb = -mu + beta * sigma
    # Higher score = more promising
    score_ucb = -mu + beta * sigma_safe
    
    # Normalize by cost to prefer cheaper evaluations.
    # Use sqrt(cost) for a softer penalty.
    cost_penalty = np.sqrt(cost_safe)
    scores = score_ucb / cost_penalty
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
