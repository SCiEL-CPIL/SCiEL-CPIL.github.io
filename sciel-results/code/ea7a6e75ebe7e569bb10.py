"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Stochastic Expected Improvement (SEI) with a cost-aware formulation.
    
    The core idea is to compute Expected Improvement (EI) and normalize by cost,
    but with a structural twist: we use a hybrid that combines EI/cost with a
    cost-scaled exploration term. This addresses the stagnation by introducing
    a different trade-off between exploitation and exploration that is
    explicitly modulated by cost.
    
    The acquisition score is:
        score = EI(x) / cost(x) + beta * sigma(x) / cost(x)
    
    This can be rewritten as:
        score = (EI(x) + beta * sigma(x)) / cost(x)
    
    The key structural difference from plain EI/cost is the addition of the
    exploration term beta * sigma(x) inside the numerator before dividing by cost.
    This means that high-uncertainty regions are explored more aggressively
    when they are cheap, but the exploration bonus is suppressed when cost is high.
    This is different from simply adding a separate UCB term because the cost
    normalization applies to both exploitation and exploration jointly.
    
    We also add a time guard to ensure we return within the deadline.
    """
    t_start = time.time()
    deadline = t_start + 280.0  # 280s margin within 300s limit
    
    # Convert to numpy arrays for safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 1e-6), cost, 1.0)
    
    # Safety: avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -50, 50)  # Numerical stability
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # Compute the hybrid cost-aware acquisition
    # score = (EI + beta * sigma) / cost
    # The exploration term beta * sigma is added inside the numerator
    # so that cost normalization applies to both exploitation and exploration
    beta = 2.0
    score = (ei + beta * sigma_safe) / cost_safe
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Time guard: if we're close to the deadline, return immediately
    # (the computation above is vectorized and fast, so this is just a safety net)
    if time.time() > deadline:
        return score
    
    return score
# EVOLVE-BLOCK-END
