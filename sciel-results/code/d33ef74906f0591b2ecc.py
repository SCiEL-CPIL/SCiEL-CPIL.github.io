"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Deterministic Upper Confidence Bound (UCB) acquisition function with
    time-decaying exploration coefficient for Bayesian optimization.
    
    Score = -mu + beta(t) * sigma, where beta(t) decreases over the 30-evaluation
    budget to shift from exploration to exploitation.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        Mean of the GP posterior at candidate points (in objective units).
    sigma : ndarray, shape (n,)
        Standard deviation of the GP posterior at candidate points (in objective units).
    f_best : float
        The lowest observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        UCB scores; higher means evaluate next.
    """
    # Estimate the number of evaluations already performed.
    # The loop starts with 5 initial points, then adds 1 per call.
    # f_best is the lowest observed value. We can't directly count calls,
    # but we can use a heuristic: the number of unique f_best values doesn't
    # help. Instead, we track state via a module-level counter.
    
    # Use a module-level counter to track how many times acquisition has been called
    if not hasattr(acquisition, '_call_count'):
        acquisition._call_count = 0
    
    acquisition._call_count += 1
    t = acquisition._call_count  # 1-based index of this call
    
    # Total budget is 30 evaluations, 5 initial + 25 acquisition calls
    # So t ranges from 1 to 25
    total_calls = 25
    t_clamped = min(t, total_calls)
    
    # Time-decaying exploration coefficient
    # beta(t) = 2.0 * (1 - t/30) as specified, but let's adjust slightly
    # Actually the directive says beta(t) = 2.0 * (1 - t/30)
    # At t=1: beta = 2.0*(1-1/30) = 2.0*0.9667 = 1.9333
    # At t=25: beta = 2.0*(1-25/30) = 2.0*0.1667 = 0.3333
    # At t=30: beta = 0
    # But we only have 25 calls, so:
    beta = 2.0 * (1.0 - t_clamped / 30.0)
    
    # Ensure beta stays positive
    beta = max(beta, 0.1)
    
    # UCB score for minimization:
    # We want to maximize the score to select the next point.
    # Higher score = more promising.
    # For minimization, we prefer low mu and high sigma (uncertainty).
    # score = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
