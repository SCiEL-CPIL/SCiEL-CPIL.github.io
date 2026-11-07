"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Hartmann 3D.
    
    Structural change: Use a Cost-Weighted Expected Improvement (CW-EI) approach
    that properly accounts for the evaluation cost in the acquisition function.
    
    The key insight is that we should maximize the expected improvement per unit
    cost, not just subtract a fixed penalty. This is the standard approach in
    cost-aware Bayesian optimization (e.g., Frazier et al. 2009).
    
    CW-EI = EI(x) / cost(x)
    
    This naturally balances exploration (via EI) with cost-awareness (via division
    by cost). Points with high expected improvement but high cost are penalized
    proportionally, while cheap points with moderate improvement are favored.
    
    Additionally, we add a small exploration bonus based on sigma to encourage
    exploring uncertain regions, weighted by cost efficiency.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill
    
    # Ensure inputs are valid
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    n = len(mu)
    
    # Basic safety: if anything is wrong, return a simple fallback
    if n == 0 or not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        return np.zeros(n, dtype=np.float64)
    
    # Clamp sigma to positive values for numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Clamp cost to positive values (cost is always >= 1.0 in this problem)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Numerical stability
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Cost-Weighted EI: EI / cost
    # This properly accounts for the cost of evaluation
    cw_ei = ei / cost_safe
    cw_ei = np.where(np.isfinite(cw_ei), cw_ei, 0.0)
    
    # Add a small exploration bonus weighted by cost efficiency
    # This encourages exploring uncertain regions, but favors cheap exploration
    exploration_weight = 0.05
    exploration_bonus = exploration_weight * sigma_safe / cost_safe
    exploration_bonus = np.where(np.isfinite(exploration_bonus), exploration_bonus, 0.0)
    
    # Combined score: CW-EI + exploration bonus
    # Higher score means more attractive to evaluate
    score = cw_ei + exploration_bonus
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e10)
    
    # Check time
    if time.time() > deadline - 2.0:
        # Fallback to simple CW-EI if running out of time
        score = cw_ei
        score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
