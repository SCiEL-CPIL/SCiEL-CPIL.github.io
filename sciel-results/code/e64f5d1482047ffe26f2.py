"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using PI with soft cost penalty.
    
    Ranks candidates by posterior probability of improving over f_best,
    divided by a soft (sqrt) cost penalty. This balances improvement
    likelihood against evaluation cost without over-penalizing slightly
    expensive points. Adds adaptive exploration bonus for early stages.
    """
    t_start = time.time()
    
    # Safety: ensure sigma is positive
    sigma_safe = np.clip(sigma, 1e-8, None)
    
    # Compute z-score: z = (f_best - mu) / sigma
    # P(f(x) < f_best) = Phi(z) where z = (f_best - mu) / sigma
    delta = f_best - mu
    z = delta / sigma_safe
    
    # Clip z for numerical stability
    z_clipped = np.clip(z, -10, 10)
    
    # Probability of improvement: P(f(x) < f_best) = Phi(z)
    p_improve = norm.cdf(z_clipped)
    
    # Ensure p_improve is in [0, 1]
    p_improve = np.clip(p_improve, 0.0, 1.0)
    
    # Soft cost penalty: use sqrt(cost) instead of linear cost
    # This avoids over-penalizing points with cost slightly above 1
    cost_safe = np.clip(cost, 1.0, None)
    soft_cost = np.sqrt(cost_safe)
    
    # Primary score: probability of improvement per unit soft cost
    score = p_improve / soft_cost
    
    # Adaptive exploration bonus:
    # - Higher uncertainty (sigma) in early stages when f_best is large
    # - Decays as f_best approaches 0 (convergence)
    # Use log1p(f_best) to get a smooth decay factor
    exploration_weight = np.minimum(1.0, np.log1p(np.maximum(f_best, 1e-6) + 1e-6))
    exploration_bonus = exploration_weight * sigma_safe / (soft_cost * 20.0)
    
    score = score + exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    if time.time() - t_start > 325:
        pass
    
    return score
# EVOLVE-BLOCK-END
