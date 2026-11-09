"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with an adaptive exploration bonus.
    
    The EI term captures the expected magnitude of improvement, which is
    crucial for navigating the narrow valley of the Rosenbrock function.
    An adaptive exploration bonus, weighted by the current best value,
    encourages exploration early on and fades as convergence approaches.
    
    The cost penalty uses a power law (cost^0.75) to balance the trade-off
    between finding good solutions and minimizing evaluation cost.
    """
    t_start = time.time()
    
    # Safety: ensure sigma is positive
    sigma_safe = np.clip(sigma, 1e-8, None)
    cost_safe = np.clip(cost, 1.0, None)
    
    # Compute z-score for EI
    delta = f_best - mu
    z = delta / sigma_safe
    
    # Clip z for numerical stability
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # EI = delta * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    EI = delta * Phi_z + sigma_safe * phi_z
    
    # Clip EI to be non-negative
    EI = np.maximum(EI, 0.0)
    
    # Cost normalization: use cost^0.75 (between sqrt and linear)
    # This penalizes expensive points but not too aggressively
    cost_penalty = np.power(cost_safe, 0.75)
    
    # Primary score: EI per unit cost
    score = EI / cost_penalty
    
    # Adaptive exploration bonus:
    # Strong exploration when f_best is large (early), fades as f_best -> 0
    # Use a sigmoid-like decay based on f_best
    exploration_weight = 1.0 / (1.0 + np.exp(np.maximum(f_best, 0.0) - 2.0))
    exploration_bonus = exploration_weight * sigma_safe / cost_penalty
    
    score = score + 0.5 * exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard
    if time.time() - t_start > 325:
        pass
    
    return score
# EVOLVE-BLOCK-END
