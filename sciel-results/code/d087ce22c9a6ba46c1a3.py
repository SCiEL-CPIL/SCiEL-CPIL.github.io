"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-adjusted Probability of Improvement (PI) acquisition function with
    adaptive exploration.
    
    Unlike Expected Improvement (EI), PI is smoother and less prone to greedy
    collapse in narrow valleys. It calculates the probability that a candidate
    is better than the current best, normalized by its evaluation cost.
    
    We add a small adaptive exploration term based on sigma to ensure we don't
    completely ignore high-uncertainty regions where PI might be low but potential
    is high. The exploration weight decreases as f_best approaches 0 (convergence).
    
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
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Base score: PI / cost
    scores = pi / cost_safe
    
    # Adaptive exploration term:
    # - Weight decreases as f_best -> 0 (we are closer to optimum)
    # - Scale by sigma to favor high-uncertainty regions
    # - Also divide by cost to maintain cost-awareness in exploration
    
    # Progress factor: 1 when f_best is large (early), 0 when f_best ~ 0 (late)
    # Use a smooth decay: 1/(1 + f_best) is in (0, 1], but we want the opposite
    # Let's use: exploration_weight = base_weight * (1 - 1/(1+f_best))
    # When f_best is large, 1/(1+f_best) ~ 0, so weight ~ base_weight
    # When f_best ~ 0, 1/(1+f_best) ~ 1, so weight ~ 0
    
    # Actually, simpler: weight = base_weight * f_best / (f_best + 1)
    # When f_best is large, weight ~ base_weight
    # When f_best ~ 0, weight ~ 0
    
    base_weight = 0.05
    exploration_weight = base_weight * (max(f_best, 0.0) / (max(f_best, 0.0) + 1.0))
    
    # Normalize sigma to [0, 1] for stable scaling
    sigma_max = np.max(sigma_safe) if len(sigma_safe) > 0 else 1.0
    sigma_norm = sigma_safe / (sigma_max + 1e-9)
    
    # Exploration term: weight * sigma_norm / cost
    exploration_term = exploration_weight * sigma_norm / cost_safe
    
    scores = scores + exploration_term
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check (computation is O(N), should be very fast)
    if time.time() - start_time > 280.0:
         return np.zeros_like(mu)
         
    return scores
# EVOLVE-BLOCK-END
