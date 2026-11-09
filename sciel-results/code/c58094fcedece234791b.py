"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-adjusted Probability of Improvement (PI) acquisition function.
    
    Unlike Expected Improvement (EI), PI is smoother and less prone to greedy
    collapse in narrow valleys. It calculates the probability that a candidate
    is better than the current best, normalized by its evaluation cost.
    
    mu, sigma: shape (2000,) GP mean and std in objective units
    f_best: lowest observed function value (scalar)
    cost: shape (2000,) evaluation cost for each candidate
    
    Returns: shape (2000,) acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    # We want P(f(x) < f_best) = Phi(z)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Cost adjustment: Divide by cost to prioritize cheaper evaluations
    # Cost is in [1, 2]. Dividing by cost effectively scales the score.
    # To make the scale more stable, we can normalize cost to [0, 1] or use it directly.
    # Using cost directly: Score = PI / cost
    # This penalizes expensive points linearly relative to their improvement probability.
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = pi / cost_safe
    
    # Add a small exploration term based on sigma to ensure we don't completely
    # ignore high-uncertainty regions where PI might be low but potential is high.
    # This is a hybrid: PI/Cost + epsilon * sigma/Cost
    # However, pure PI/Cost is a strong structural change. Let's stick to PI/Cost
    # but add a tiny noise floor to avoid exact zeros.
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check (computation is O(N), should be very fast)
    if time.time() - start_time > 280.0:
         return np.zeros_like(mu)
         
    return scores
# EVOLVE-BLOCK-END
